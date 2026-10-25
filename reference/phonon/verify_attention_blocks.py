"""Weightless identical-operand native/Torch2.7.1 SDPA block-rounding gate.

No model weights, native frontend, frozen fixtures or tolerance updates. Tests
full/partial second blocks, >2 blocks, and causal batched prefixes. The native
shared library is development-only and exports test primitives, never a runtime.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LENGTHS = [140,285,512,513,514,515,516,767,768,1023,1024,1025,1537,2049]
PREFILL_LENGTHS = [513,514,515,516,767,768,1023,1024,1025,1537]


def interleave(x):
    return np.stack((x[...,:32],x[...,32:]),axis=-1).reshape(x.shape)


def packed(x, reorder=False):
    value = x.numpy()
    if reorder:
        value = interleave(value)
    return value[0].transpose(1,0,2).reshape(value.shape[-2],512).copy()


def verify():
    assert torch.__version__.split('+')[0] == '2.7.1'
    torch.set_num_threads(1)
    torch.manual_seed(28)
    output = ROOT/'.build/phonon-attention-probe.dylib'
    output.parent.mkdir(exist_ok=True)
    sources = ['phonon_transformer.c','phonon_pack.c','phonon_q8.c','phonon_math.c','sha256.c']
    subprocess.run(['xcrun','clang','-std=c17','-O2','-Isrc','-dynamiclib',
                    '-DPHONON_PACK_TEST_API', *['src/'+s for s in sources],
                    '-framework','Accelerate','-o',str(output)],cwd=ROOT,check=True)
    lib = C.CDLL(str(output))
    single = lib.phonon_attention_test
    single.argtypes = [C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p]
    batch = lib.phonon_prefill_attention_test
    batch.argtypes = [C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p]
    records = []
    for n in LENGTHS:
        q,k,v = torch.randn(1,8,1,64),torch.randn(1,8,n,64),torch.randn(1,8,n,64)
        expected = torch.nn.functional.scaled_dot_product_attention(q,k,v).numpy().reshape(8,64)
        nq,nk,nv = interleave(q.numpy()).reshape(8,64).copy(),packed(k,True),packed(v)
        actual = np.empty((8,64),dtype='<f4')
        for h in range(8):
            offset = h*64*4
            assert single(nq.ctypes.data+offset,nk.ctypes.data+offset,nv.ctypes.data+offset,
                          n,512,actual.ctypes.data+offset)
        np.testing.assert_array_equal(actual,expected,err_msg=f'single query length{n}')
        records.append({'kind':'single','length':n,'exact':True})
        print(f'single query length={n} bit exact',flush=True)
    for n in PREFILL_LENGTHS:
        q,k,v = [torch.randn(1,8,n,64) for _ in range(3)]
        expected = packed(torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True))
        projected = np.zeros((n,1536),dtype='<f4')
        projected[:,:512] = packed(q,True)
        nk,nv,actual = packed(k,True),packed(v),np.empty((n,512),dtype='<f4')
        assert batch(projected.ctypes.data,nk.ctypes.data,nv.ctypes.data,n,actual.ctypes.data)
        np.testing.assert_array_equal(actual,expected,err_msg=f'causal prefill length{n}')
        records.append({'kind':'prefill','length':n,'exact':True})
        print(f'causal prefill length={n} bit exact',flush=True)
    manifest = {'torch':torch.__version__,'records':records,
                'operator_provenance':json.loads((ROOT/'reference/phonon/operator-provenance.json').read_text()),
                'native_sources_sha256':{s:hashlib.sha256((ROOT/'src'/s).read_bytes()).hexdigest() for s in sources},
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    verify()
