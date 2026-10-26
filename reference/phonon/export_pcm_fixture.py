"""Independent actual GGUF q8/BF16 PCM bridge; frozen inputs only for F32 probes.

Run with locked uv project. Never writes frozen fixtures or tolerance budgets.
"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
import torch
from oracle import load, REV, MIMI_REV, HF_REV

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / '.build' / 'phonon-pcm-q8-v1.bin'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    torch.set_num_threads(1)
    _, _, codec, proc = load('q8')
    assert proc.fade_in_duration == 0 and codec.config.frame_size == 1920
    frozen_manifest = json.loads((HERE / 'fixtures' / 'files-manifest.json').read_text())
    records, sources = [], []
    with torch.inference_mode():
        for name in ('phonon-hello-q8.npz', 'phonon-hello-q8-natural.npz',
                     'mimi-temporal-263.npz', 'mimi-uneven-263-257.npz'):
            path = HERE / 'fixtures' / name
            assert digest(path) == frozen_manifest['files'][name]['sha256']
            sources.append({'path': str(path.relative_to(HERE)), 'sha256': digest(path),
                            'used': 'frozen latent INPUTS; q8 hello PCM anchors only'})
            with np.load(path) as frozen:
                latents = torch.from_numpy(frozen['latents'].copy())
                lengths = frozen['lengths'].tolist() if 'lengths' in frozen else [latents.shape[1]]
                mask = torch.arange(latents.shape[1])[None, :] < torch.tensor(lengths)[:, None]
                batch_final = proc.decode(latents, mask)
                for row, length in enumerate(lengths):
                    latent = latents[row:row+1, :length]
                    raw = codec.decode(latent, return_dict=True).audio_values[0, 0].float()
                    final = proc.decode(latent)[0]
                    np.testing.assert_array_equal(raw.numpy(), final.numpy())
                    np.testing.assert_allclose(batch_final[row].numpy(), final.numpy(), rtol=0, atol=2e-7)
                    assert raw.numel() == length * 1920
                    if name.startswith('phonon-hello'):
                        np.testing.assert_array_equal(raw.numpy(), frozen['raw_pcm'][0])
                        np.testing.assert_array_equal(final.numpy(), frozen['final_pcm'][0])
                    records.append((latent.numpy(), raw.numpy(), final.numpy()))
                    if name == 'mimi-temporal-263.npz':
                        # Independent HF cached transformer + causal overlap SEANet.
                        hidden = codec.decoder.upsample(codec.latent_proj(latent.transpose(1, 2))).transpose(1, 2)
                        for chunks in ([1] * length, [1, 7, 113, 142]):
                            cache, offset, features, output = None, 0, [], []
                            for frames in chunks:
                                end = offset + frames * 16
                                decoded = codec.decoder.decoder_transformer(hidden[:, offset:end], past_key_values=cache, use_cache=True, return_dict=True)
                                cache = decoded.past_key_values
                                features.append(decoded.last_hidden_state)
                                all_features = torch.cat(features, dim=1)
                                start = max(0, offset - 250)
                                segment = codec.decoder.decoder_seanet(all_features[:, start:end].transpose(1, 2))[0, 0]
                                output.append(segment[(offset-start)*120:])
                                offset = end
                            pcm = torch.cat(output).float().numpy()
                            np.testing.assert_allclose(pcm, raw.numpy(), rtol=0, atol=1e-5)
                            records.append((latent.numpy(), pcm, pcm.copy()))
                    if name == 'phonon-hello-q8-natural.npz':
                        # Tiny, actual retained EOS tail and zero latent tail are
                        # independent decode inputs, never fabricated flush frames.
                        for piece in (latent[:, :1], latent[:, :2], latent[:, -3:],
                                      torch.cat((latent[:, :1], torch.zeros_like(latent[:, :2])), dim=1)):
                            value = codec.decode(piece, return_dict=True).audio_values[0, 0].float().numpy()
                            records.append((piece.numpy(), value, proc.decode(piece)[0].numpy()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('wb') as stream:
        stream.write(b'PHPCM1\0\0')
        stream.write(struct.pack('<I', len(records)))
        for latent, raw, final in records:
            stream.write(struct.pack('<II', latent.shape[1], raw.size))
            for value in (latent, raw, final):
                stream.write(np.asarray(value, dtype='<f4').tobytes())
    manifest = {'schema_version': 1, 'model_revision': REV, 'mimi_revision': MIMI_REV,
                'transformers_revision': HF_REV, 'torch': torch.__version__,
                'weights': 'oracle.load(q8): all 209 GGUF tensors, BF16 Mimi expansion',
                'source_inputs': sources, 'records': len(records),
                'seanet_layers': str(codec.decoder.decoder_seanet.layers),
                'postprocess': 'HF processor: retain mask-prefix frames *1920; fade_in_duration=0; no gain/limiter',
                'delay_flush': 'causal left padding/right crop per operator; 1920 samples/frame immediately; zero codec flush; EOS tail upstream',
                'processor_settings': json.loads((__import__('oracle').ASSETS / 'processor_config.json').read_text()),
                'pins_sha256': digest(HERE / 'pins.json'),
                'source_hashes_sha256': digest(HERE / 'source-hashes.json'),
                'budgets_sha256': digest(HERE / 'budgets.json'),
                'frozen_manifest_sha256': digest(HERE / 'fixtures' / 'files-manifest.json'),
                'exporter_sha256': digest(Path(__file__)), 'bridge_sha256': digest(OUT)}
    OUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{OUT}: {len(records)} independent q8 raw/final PCM records')


if __name__ == '__main__':
    export()
