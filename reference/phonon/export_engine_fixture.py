"""Independent resident-engine oracle: pinned Rust English/prepare + HF q8/BF16.

No native frontend or engine outputs define expected IDs/audio. The existing
independent planner reference supplies split/fit policy; Rust executes prompt
preparation. Replayable xoshiro/Box-Muller draws bridge PRNG differences only.
Never changes frozen fixtures or budgets. Run using the locked phonon uv project.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import numpy as np
import torch
from tokenizers import Tokenizer
from safetensors.torch import load_file, save_file
from oracle import load, build_inputs, ASSETS, ROOT, REV, HF_REV, MIMI_REV

OUT = ROOT / '.build/phonon-engine-v1.bin'
NORMALIZER = ROOT / '.build/normalization-oracle-target/debug/phonon-english-normalization-oracle'
PREPARE = ROOT / '.build/phonon-prompt-prepare-oracle/prepare'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(text):
    row = subprocess.run([str(PREPARE)], input=(text.encode().hex()+'\n').encode(),
                         stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    text, tail = row.rsplit(':', 1)
    return bytes.fromhex(text).decode(), int(tail)


class Noise:
    """Independent implementation of documented request RNG, not Torch seed parity."""
    mask = (1 << 64)-1

    def __init__(self, seed):
        self.state = []
        for _ in range(4):
            seed = (seed + 0x9e3779b97f4a7c15) & self.mask
            z = seed
            z = ((z ^ (z >> 30))*0xbf58476d1ce4e5b9) & self.mask
            z = ((z ^ (z >> 27))*0x94d049bb133111eb) & self.mask
            self.state.append(z ^ (z >> 31))
        self.spare = None
        self.draws = []

    def rot(self, x, n):
        return ((x << n) | (x >> (64-n))) & self.mask

    def uniform(self):
        s = self.state
        result = self.rot((s[1]*5) & self.mask, 7)*9 & self.mask
        t = s[1] << 17 & self.mask
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]
        s[2] ^= t; s[3] = self.rot(s[3], 45)
        return ((result >> 11)+.5)/9007199254740992.0

    def next(self):
        if self.spare is not None:
            value, self.spare = self.spare, None
        else:
            radius = math.sqrt(-2*math.log(self.uniform()))
            angle = 6.283185307179586476925286766559*self.uniform()
            value = np.float32(radius*math.cos(angle))
            self.spare = np.float32(radius*math.sin(angle))
        self.draws.append(float(value))
        return value

    def randn(self, *shape, **kw):
        assert shape == (1, 32), shape
        return torch.tensor([self.next() for _ in range(32)], dtype=kw['dtype'],
                            device=kw['device']).reshape(shape)


def export():
    torch.set_num_threads(1)
    tok, model, codec, proc = load('q8')
    raw_tokenizer = Tokenizer.from_file(str(ASSETS/'tokenizer.json'))
    assert raw_tokenizer.encode('.!...?', add_special_tokens=False).ids == [660,663,1367,1293,691]
    # Reuse the independently source-checked planner functions without invoking
    # its CLI/vector writer. No native implementation is loaded by this script.
    planner = ROOT / 'tools/generate_prompt_plan_vectors.py'
    names = {'split', 'chunk', 'cut', 'fit'}
    functions = [node for node in ast.parse(planner.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    env = {'prep': prepare, 'enc': lambda s: raw_tokenizer.encode(s, add_special_tokens=False).ids,
           'tok': raw_tokenizer, 'math': math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(planner), 'exec'), env)
    source_pins = json.loads((ROOT/'tools/prompt_plan_source_hashes.json').read_text())
    for name, digest in source_pins['files'].items():
        assert sha(ROOT/'.build/xn-ptts/repo'/name) == digest
    original_voice = load_file(str(ASSETS/'default-voice.safetensors'))['speaker_wavs']
    alternate = original_voice[:, :, :32].contiguous()
    voice_path = ROOT/'.build/phonon-engine-short-voice.safetensors'
    save_file({'speaker_wavs': alternate}, str(voice_path), metadata={'sig':'ba7a5a45','epoch':'300'})
    cases = [
        ('Hello!', 123, 0., original_voice),
        ('I have 2 cafés.', 456, .003, original_voice),
        ('I paid $12.50!', 789, -.003, alternate),
        ('Hello! ' * 27, 123, 0., original_voice),
        ('hi ' * 220, 27, 0., alternate),
    ]
    records, metadata = [], []
    original_randn = torch.randn
    with torch.inference_mode():
        for record_index, (text, seed, delta, voice_tensor) in enumerate(cases):
            normalized = subprocess.run([str(NORMALIZER)], input=(text+'\n').encode(),
                                        stdout=subprocess.PIPE, check=True).stdout.decode().rstrip('\n')
            chunks = env['fit']([env['chunk'](s) for s in env['split'](normalized, 50)], 200)
            noise, pcms, chunk_meta, traces = Noise(seed), [], [], []
            original_sample = model.sample_latent
            def sample(condition, *args, **kwargs):
                value = original_sample(condition, *args, **kwargs)
                traces.append((condition.detach().cpu().numpy().reshape(-1), value.detach().cpu().numpy().reshape(-1)))
                return value
            model.sample_latent = sample
            norm_regression = []
            def capture_norm(_module, args, _output):
                if record_index == 3 and len(traces) == 109:
                    norm_regression.append(args[0].detach().cpu().float().numpy().reshape(-1).copy())
            norm_hook = model.model.norm.register_forward_hook(capture_norm)
            model.model.config.duration_delta = delta
            # model and outer config are shared by HF in this pinned build.
            model.config.duration_delta = delta
            try:
                torch.randn = noise.randn
                for c in chunks:
                    prepared, tail = prepare(c['text'])
                    assert raw_tokenizer.encode(prepared, add_special_tokens=False).ids == c['ids']
                    voice = voice_tensor[0].T.float()
                    inputs = build_inputs(tok, model, proc, prepared, voice)
                    # The converted HF tokenizer adds punctuation rewrites which are
                    # NOT the pinned Rust raw Unigram frontend. Use independent
                    # raw HF tokenizers IDs; keep HF model/voice assembly intact.
                    ids = [model.config.audio_prompt_token_id]*len(voice) + c['ids'] + [model.config.audio_bos_token_id]
                    inputs['input_ids'] = torch.tensor([ids], dtype=torch.long)
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                    inputs['frames_after_eos'] = tail
                    trace_start = len(traces)
                    output = model.generate(**inputs, max_new_tokens=min(c['frames'],4096),
                                            return_dict_in_generate=True, output_scores=True)
                    for j, logit in enumerate(output.scores):
                        condition, latent = traces[trace_start+j]
                        traces[trace_start+j] = np.concatenate((condition,latent,[float(logit[0])])).astype('<f4')
                    count = int(output.audio_latents_mask.sum())
                    assert count == output.audio_latents.shape[1]
                    pcm = proc.decode(output.audio_latents, output.audio_latents_mask)[0].numpy()
                    assert pcm.size == count*1920
                    pcms.append(pcm)
                    chunk_meta.append({'prepared': prepared, 'ids': c['ids'], 'tail': tail,
                                       'budget': c['frames'], 'frames': count,
                                       'last_eos': float(output.scores[-1][0])})
            finally:
                torch.randn = original_randn
                model.sample_latent = original_sample
                norm_hook.remove()
            if record_index == 3:
                import re
                vector = (ROOT/'tests/phonon_layer_norm_regression.inc').read_text().split('= {',1)[1]
                bits = np.array([int(x,16) for x in re.findall(r'0x([0-9a-f]{8})U',vector)],dtype='<u4')
                assert len(norm_regression) == 1 and np.array_equal(norm_regression[0].view('<u4'),bits)
                scalar = torch.nn.functional.layer_norm(torch.from_numpy(norm_regression[0]),[512],eps=1e-5)
                assert int(scalar.numpy().view('<u4')[0]) == 0x3f31cd52
            pcm = np.concatenate(pcms).astype('<f4')
            records.append((text, seed, delta, 32 if voice_tensor is alternate else 125,
                            len(chunks), pcm, np.array(noise.draws, dtype='<f4'), np.stack(traces)))
            metadata.append({'text': text, 'normalized': normalized, 'seed': seed, 'delta': delta,
                             'chunks': chunk_meta, 'samples': pcm.size})
            print(f'engine oracle voice={voice_tensor.shape[-1]} chunks={len(chunks)} samples={pcm.size}', flush=True)
    with OUT.open('wb') as f:
        f.write(b'PHENG2\0\0'); f.write(struct.pack('<I', len(records)))
        for text, seed, delta, voice_frames, chunks, pcm, draws, trace in records:
            raw = text.encode()
            f.write(struct.pack('<IQfIIII', len(raw), seed, delta, voice_frames, chunks, pcm.size, draws.size))
            f.write(raw); f.write(draws.tobytes()); f.write(trace.tobytes()); f.write(pcm.tobytes())
    manifest = {'model_revision': REV, 'mimi_revision': MIMI_REV, 'transformers_revision': HF_REV,
                'weights': 'all 209 GGUF tensors: 36 Q8 +173 BF16; oracle.load(q8)',
                'effective_config': {'temperature': .8, 'snapshot_temperature': .3,
                                     'eos_threshold': 4, 'num_flow_steps': 1, 'min_frames_before_eos': 0},
                'normalizer': str(NORMALIZER), 'normalizer_sha256': sha(NORMALIZER),
                'tokenizer_sha256': sha(ASSETS/'tokenizer.json'), 'prepare_sha256': sha(PREPARE), 'planner_reference_sha256': sha(planner),
                'source_pins': source_pins, 'exporter_sha256': sha(Path(__file__)),
                'bridge_sha256': sha(OUT), 'voice_sha256': sha(voice_path), 'records': metadata}
    OUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    export()
