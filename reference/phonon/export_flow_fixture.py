"""Export the frozen Q8 prompt/backbone trace to a tiny C-readable fixture."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
import sys
root = Path(__file__).parent
sys.path.insert(0, str(root))
import oracle
with np.load(root / "fixtures" / "phonon-hello-q8.npz") as fixture:
    ids = fixture["input_ids"].reshape(-1).astype("<i4")
    voice = fixture["voice_prompt_latents"].astype("<f4")
    expected = fixture["backbone_output"].reshape(-1, 512).astype("<f4")
    latents = fixture["latents"].reshape(-1, 32).astype("<f4")
    voice_projection = fixture["voice_projection"].reshape(-1, 512).astype("<f4")
    if voice.shape != (125, 512) or expected.shape[0] < ids.size or latents.shape[0] != expected.shape[0] - ids.size + 1:
        raise ValueError("unexpected pinned prefix fixture dimensions")
    output = Path(".build/phonon-flow-prefix-v1.bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    _, model, _, _ = oracle.load("q8")
    with __import__("torch").inference_mode():
        bos_embedding = model.model.embed_audio_latents(model.model.bos_latent).float().cpu().numpy().astype("<f4")
    with output.open("wb") as stream:
        stream.write(struct.pack("<8sIIII", b"PHFLOW1\0", ids.size, voice.shape[0], 512, expected.shape[0]))
        stream.write(ids.tobytes())
        stream.write(voice.tobytes())
        stream.write(voice_projection.tobytes())
        stream.write(bos_embedding.tobytes())
        stream.write(latents.tobytes())
        stream.write(expected.tobytes())
print(output)

conditions = []
with __import__("torch").inference_mode():
    for delta in (0.003, -0.003):
        model.config.duration_delta = delta
        conditions.append(model.model.get_condition_embeds("cpu").float().cpu().numpy())
condition_output = Path(".build/phonon-flow-condition-v1.bin")
with condition_output.open("wb") as stream:
    stream.write(struct.pack("<8sII", b"PHCOND1\0", 2, 512))
    stream.write(np.asarray(conditions, dtype="<f4").tobytes())
print(condition_output)
manifest = {
    "schema_version": 1,
    "model_revision": oracle.REV,
    "transformers_revision": oracle.HF_REV,
    "source_fixture": "phonon-hello-q8.npz",
    "source_fixture_sha256": hashlib.sha256((root / "fixtures" / "phonon-hello-q8.npz").read_bytes()).hexdigest(),
    "native_prefix_binary": str(output),
    "condition_binary": str(condition_output),
    "condition_deltas": [0.003, -0.003],
}
manifest_path = Path(".build/phonon-flow-fixture-v1.json")
manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
print(manifest_path)
