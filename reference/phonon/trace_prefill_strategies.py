"""Read-only Torch B1/full versus incremental-prefix replay diagnostics.

This does not replace, narrow, regenerate, or recalibrate the frozen native gate.
All strategies use the same pinned model, exact saved raw noise, and B1 head.
"""
from pathlib import Path
import sys
import ctypes as C
import subprocess
import tempfile
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle
from verify_native_generation_trace import read_trace, within, STAGE_BUDGETS, PCM_BUDGETS


def native_prefix(embedded):
    scratch = Path(tempfile.mkdtemp(prefix="phonon-prefix-trace-"))
    wrapper = scratch / "prefix.c"
    wrapper.write_text(r'''#include "phonon_transformer.h"
#include <string.h>
int trace_prefix(const float *input, size_t length, float *output, float *keys, float *values) {
    PhononPack *pack = NULL; PhononKvCache cache = {0};
    PhononTensorView storage[9][8]; PhononFlowLayerWeights weights[9]; int ok = 0;
    if (!phonon_pack_open(".build/phonon-native-f6acc8f43cb296be-v1.kpack", &pack) ||
        !phonon_flow_load_layers(pack, storage, weights) || !phonon_kv_cache_init(&cache,length)) goto end;
    if (!phonon_flow_stack_prefill(weights,&cache,input,length,10000.0f,output) ||
        !phonon_flow_output_norm(pack,output)) goto end;
    memcpy(keys,cache.keys,9*length*512*sizeof(float));
    memcpy(values,cache.values,9*length*512*sizeof(float)); ok=1;
end: phonon_kv_cache_destroy(&cache); phonon_pack_close(pack); return ok;
}
''')
    root = HERE.parents[1]
    library = scratch / "prefix.dylib"
    subprocess.run(["clang", "-std=c17", "-O0", "-dynamiclib", "-I" + str(root / "src"),
                    str(wrapper), *(str(root / "src" / name) for name in
                    ("phonon_transformer.c", "phonon_q8.c", "phonon_pack.c", "sha256.c", "phonon_math.c")),
                    "-framework", "Accelerate", "-o", str(library)], check=True)
    lib = C.CDLL(str(library))
    lib.trace_prefix.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p, C.c_void_p]
    inputs = embedded.numpy().copy()
    length = inputs.shape[1]
    output = np.empty(512, dtype="float32")
    keys = np.empty((9, length, 512), dtype="float32")
    values = np.empty_like(keys)
    assert lib.trace_prefix(inputs.ctypes.data, length, output.ctypes.data, keys.ctypes.data, values.ctypes.data)
    return output, keys, values


def main():
    _, model, codec, _ = oracle.load("q8")
    torch.set_num_threads(1)
    with np.load(HERE / "fixtures/phonon-hello-q8-natural.npz") as f:
        frozen = {key: f[key].copy() for key in f.files}
    ids = torch.from_numpy(frozen["input_ids"])
    voice = torch.from_numpy(frozen["voice_prompt_latents"])
    noise = torch.from_numpy(frozen["noise_draws"])
    results = {}
    with torch.inference_mode():
        embedded = model.model.embed_tokens(ids)
        voice_embedded = model.model.audio_prompt_proj(voice)
        embedded[ids == model.config.audio_prompt_token_id] = voice_embedded.reshape(-1, 512)
        embedded[ids == model.config.audio_bos_token_id] = model.model.embed_audio_latents(model.model.bos_latent)
        for name in ("full", "incremental"):
            cache = None
            if name == "full":
                out = model.model(inputs_embeds=embedded, use_cache=True)
                cache, state = out.past_key_values, out.last_hidden_state[:, -1]
                native_state, native_keys, native_values = native_prefix(embedded)
                print(f"Native matched-embedding full-prefix last condition max_abs={np.abs(native_state-state.numpy()[0]).max():.9g}")
                for layer in range(9):
                    key, value = cache.layers[layer].keys, cache.layers[layer].values
                    key = key[0].transpose(0, 1).numpy()
                    # HF converter maps raw interleaved Q/K into rotate_half layout.
                    interleaved = np.stack((key[..., :32], key[..., 32:]), axis=-1).reshape(-1, 512)
                    value = value[0].transpose(0, 1).numpy().reshape(-1, 512)
                    print(f"  prefix cache layer{layer}: key max_abs={np.abs(native_keys[layer]-interleaved).max():.9g}; "
                          f"value max_abs={np.abs(native_values[layer]-value).max():.9g}")
            else:
                for i in range(embedded.shape[1]):
                    out = model.model(inputs_embeds=embedded[:, i:i+1], past_key_values=cache, use_cache=True)
                    cache, state = out.past_key_values, out.last_hidden_state[:, -1]
            conditions, integration, logits = [], [], []
            for frame in range(32):
                conditions.append(state.clone())
                logits.append(model.eos_head(state).clone())
                normalized = model.sample_latent(state, noise[frame] * np.sqrt(0.8), 1)
                integration.append(normalized.clone())
                if frame != 31:
                    feedback = model.model.embed_audio_latents(normalized).unsqueeze(1)
                    out = model.model(inputs_embeds=feedback, past_key_values=cache, use_cache=True)
                    cache, state = out.past_key_values, out.last_hidden_state[:, -1]
            conditions = torch.cat(conditions).numpy()
            integration_tensor = torch.stack(integration, dim=1)
            latents = integration_tensor * model.latent_std + model.latent_mean
            pcm = codec.decode(latents, return_dict=True).audio_values[0, 0].numpy()
            results[name] = (conditions, integration_tensor.numpy().reshape(-1, 32),
                             torch.cat(logits).numpy().reshape(-1), latents.numpy()[0], pcm)
        conditions, draws, native, _, _ = read_trace()
        c = torch.from_numpy(conditions.copy())
        n = torch.from_numpy(draws.copy()) * np.sqrt(.8)
        b1 = torch.cat([model.sample_latent(c[i:i+1], n[i:i+1], 1) for i in range(32)]).numpy()
        b32 = model.sample_latent(c, n, 1).numpy()
        print(f"Same-input head B1-vs-B32 max_abs={np.abs(b1-b32).max():.9g}; "
              f"native-vs-B1 max_abs={np.abs(native-b1).max():.9g}")
    anchors = (frozen["flow_condition"].reshape(-1, 512), frozen["flow_integration"].reshape(-1, 32),
               frozen["eos_logits"].reshape(-1), frozen["latents"][0], frozen["raw_pcm"][0])
    for name, arrays in results.items():
        print(name, "versus frozen:")
        for stage, actual, expected in zip(("flow_condition", "flow_integration", "eos_logits", "latents"), arrays[:4], anchors[:4]):
            ok, maximum, bad = within(actual, expected, STAGE_BUDGETS[stage])
            print(f"  {stage}: pass={ok}; max_abs={maximum:.9g}; over={bad}/{actual.size}")
        error = arrays[4] - anchors[4]
        nrmse = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(anchors[4]**2)))
        maximum = float(np.abs(error).max())
        print(f"  PCM: nrmse={nrmse:.9g}; max_abs={maximum:.9g}; "
              f"pass={nrmse <= PCM_BUDGETS['normalized_rmse_max'] and maximum <= PCM_BUDGETS['max_abs']}")
    if not all(np.array_equal(a, b) for a, b in zip(results["full"], anchors)):
        raise SystemExit("Full-prefix reconstruction is not exact frozen replay; strategy diagnosis is inconclusive.")
    print("Full-prefix strategy exactly replays frozen outputs; no fixtures changed.")


if __name__ == "__main__":
    main()
