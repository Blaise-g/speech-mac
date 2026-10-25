"""Independent Torch 2.7.1 q8/BF16 SEANet operator seams; no frozen writes."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
import torch
from oracle import load, REV, MIMI_REV, HF_REV

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / '.build' / 'phonon-conv-q8-v1.bin'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    torch.set_num_threads(1)
    _, _, codec, _ = load('q8')
    layers = codec.decoder.decoder_seanet.layers
    records = []
    backend = []
    generator = torch.Generator().manual_seed(260519)
    with torch.inference_mode():
        # Independent modules with real BF16-expanded weights. Their inputs are
        # deliberately asymmetric and include negatives, impulses and zero tail.
        for index, layer in enumerate(layers):
            residual = hasattr(layer, 'block')
            if not residual and not hasattr(layer, 'conv'):
                continue
            channels = layer.block[1].conv.in_channels if residual else layer.conv.in_channels
            x = torch.randn((1, channels, 17), generator=generator) * 0.31
            x[:, :, 0] += torch.arange(channels)[None, :] * 0.002
            x[:, :, -3:] = 0
            kind = 2 if residual else int(isinstance(layer.conv, torch.nn.ConvTranspose1d))
            expected = layer(x)
            # Profile actual operator dispatch; shape-equivalent assumptions are
            # insufficient to identify Torch's accumulation/bias placement.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                layer(x)
            ops = sorted({event.key for event in prof.key_averages()
                          if 'conv' in event.key or 'elu' in event.key or event.key == 'aten::add'})
            backend.append({'layer': index, 'kind': kind, 'operators': ops})
            if not residual:
                stride = layer.conv.stride[0]
                weights = [layer.conv.weight, layer.conv.bias]
            else:
                stride = 1
                weights = [layer.block[1].conv.weight, layer.block[1].conv.bias,
                           layer.block[3].conv.weight, layer.block[3].conv.bias]
            # Whole module forward remains the oracle; no native output used.
            records.append((index, kind, stride, weights, x, expected))
    with OUT.open('wb') as stream:
        stream.write(b'PHCONV1\0')
        stream.write(struct.pack('<I', len(records)))
        for index, kind, stride, weights, x, expected in records:
            stream.write(struct.pack('<7I', index, kind, stride, x.shape[1], expected.shape[1],
                                     x.shape[2], len(weights)))
            for weight in weights:
                shape = tuple(weight.shape)
                stream.write(struct.pack('<4I', len(shape), *(shape + (0,) * (3 - len(shape)))))
                stream.write(weight.to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2').tobytes())
            for array in (x, expected):
                stream.write(array[0].T.contiguous().numpy().astype('<f4').tobytes())
    manifest = {'schema_version': 1, 'model_revision': REV, 'mimi_revision': MIMI_REV,
                'transformers_revision': HF_REV, 'torch': torch.__version__,
                'weights': 'oracle.load(q8): all 209 published tensors, BF16 Mimi expansion',
                'input': 'seed 260519, asymmetric random/impulse with three-position zero tail',
                'positions': 17, 'ratios': [6, 5, 4], 'backend': backend,
                'budgets': 'unchanged component engineering target rtol=1e-3, atol=1e-5; gamma_n for 3584 terms',
                'pins_sha256': digest(HERE / 'pins.json'),
                'budgets_sha256': digest(HERE / 'budgets.json'),
                'source_hashes_sha256': digest(HERE / 'source-hashes.json'),
                'exporter_sha256': digest(Path(__file__)), 'bridge_sha256': digest(OUT)}
    OUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    export()
