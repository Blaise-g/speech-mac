"""Independent GGUF-BF16 Mimi feature bridge; never alters frozen fixtures/budgets.

Temporal fixture outputs use original F32 Mimi weights, so only their pinned
latent INPUTS are reused. Expected features below come from oracle.load('q8').
"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
import torch
from oracle import load, REV, MIMI_REV, HF_REV

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / '.build' / 'phonon-mimi-q8-v1.bin'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    _, _, codec, _ = load('q8')
    frozen_manifest = json.loads((HERE / 'fixtures' / 'files-manifest.json').read_text())
    records = []
    inputs = []
    with torch.inference_mode():
        for name in ('phonon-hello-q8.npz', 'phonon-hello-q8-natural.npz',
                     'mimi-temporal-263.npz', 'mimi-uneven-263-257.npz'):
            path = HERE / 'fixtures' / name
            if digest(path) != frozen_manifest['files'][name]['sha256']:
                raise RuntimeError(f'frozen fixture hash mismatch: {name}')
            inputs.append({'path': str(path.relative_to(HERE)), 'sha256': digest(path),
                           'used': 'latents and lengths only; frozen features only for q8 hello'})
            with np.load(path) as frozen:
                latents = torch.from_numpy(frozen['latents'].copy())
                lengths = frozen['lengths'].tolist() if 'lengths' in frozen else [latents.shape[1]]
                batched = None
                if len(lengths) > 1:
                    hidden_batch = codec.decoder.upsample(codec.latent_proj(latents.transpose(1, 2))).transpose(1, 2)
                    mask = torch.arange(hidden_batch.shape[1])[None, :] < torch.tensor(lengths)[:, None] * 16
                    batched = codec.decoder.decoder_transformer(
                        hidden_batch, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
                for row, length in enumerate(lengths):
                    latent = latents[row:row + 1, :length]
                    hidden = codec.decoder.upsample(codec.latent_proj(latent.transpose(1, 2))).transpose(1, 2)
                    transformer = codec.decoder.decoder_transformer
                    whole = transformer(hidden, use_cache=False, return_dict=True).last_hidden_state
                    expected = [whole]
                    if batched is not None:
                        batch_row = batched[row:row + 1, :length * 16]
                        if (whole - batch_row).abs().max().item() > 1e-5:
                            raise RuntimeError('independent q8 masked uneven batch changed')
                        expected.append(batch_row)
                    if name == 'mimi-temporal-263.npz':
                        for chunks in ([1] * 263, [1, 7, 113, 142]):
                            cache, offset, pieces = None, 0, []
                            for frames in chunks:
                                end = offset + frames * 16
                                out = transformer(hidden[:, offset:end], past_key_values=cache,
                                                  use_cache=True, return_dict=True)
                                cache = out.past_key_values
                                pieces.append(out.last_hidden_state)
                                offset = end
                            partitioned = torch.cat(pieces, dim=1)
                            if (whole - partitioned).abs().max().item() > 1e-5:
                                raise RuntimeError('independent q8 cache partition changed')
                            expected.append(partitioned)
                    if name.startswith('phonon-hello'):
                        np.testing.assert_array_equal(whole.numpy(), frozen['mimi_features'])
                    records.append((latent.numpy(), hidden.numpy(), expected))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('wb') as stream:
        stream.write(b'PHMIMI1\0')
        stream.write(struct.pack('<I', len(records)))
        for latent, hidden, expected in records:
            stream.write(struct.pack('<II', latent.shape[1], len(expected)))
            for array in (latent, hidden, *(item.numpy() for item in expected)):
                stream.write(np.asarray(array, dtype='<f4').tobytes())
    manifest = {'schema_version': 1, 'weights': 'q8: all 209 published GGUF tensors; 44 BF16 Mimi tensors',
                'model_revision': REV, 'mimi_revision': MIMI_REV, 'transformers_revision': HF_REV,
                'context': 250, 'attention': codec.config._attn_implementation,
                'torch': torch.__version__, 'source_inputs': inputs,
                'pins_sha256': digest(HERE / 'pins.json'),
                'gguf_sha256': next(item['sha256'] for item in json.loads((HERE / 'pins.json').read_text())['files']
                                    if item['path'] == 'model.q8.gguf'),
                'frozen_manifest_sha256': digest(HERE / 'fixtures' / 'files-manifest.json'),
                'source_hashes_sha256': digest(HERE / 'source-hashes.json'),
                'budgets_sha256': digest(HERE / 'budgets.json'),
                'exporter_sha256': digest(Path(__file__)), 'bridge_sha256': digest(OUT),
                'records': [{'frames': int(r[0].shape[1]), 'expectations': len(r[2])} for r in records]}
    OUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    export()
