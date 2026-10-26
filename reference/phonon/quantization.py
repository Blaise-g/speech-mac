"""Expand every tensor in the pinned mixed Q8_0/BF16 GGUF for HF reference execution."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path(os.environ.get('PHONON_MODEL_DIR','models/phonon-40m-en'))
PINS=json.loads((Path(__file__).parent/'pins.json').read_text())
GGUF_PIN=next(item for item in PINS['files'] if item['path']=='model.q8.gguf')
CONVERTER_VERSION='gguf-full-tensor-expansion-v2'
OUT=ROOT/'.build'/f"reference-phonon-q8-{GGUF_PIN['sha256'][:16]}-{CONVERTER_VERSION}"

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def tensor_types(reader):
    return dict(sorted(Counter(t.tensor_type.name for t in reader.tensors).items()))

def prepare_q8_checkpoint():
    from verify_assets import verify
    verify()
    gguf_path=ASSETS/'model.q8.gguf'
    observed_hash=sha256(gguf_path)
    if observed_hash!=GGUF_PIN['sha256']:
        raise RuntimeError('Q8 GGUF hash differs from committed source pin')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'model.safetensors'; marker=OUT/'derived-manifest.json'
    reader=GGUFReader(str(gguf_path))
    if len(reader.tensors)!=209 or tensor_types(reader)!={'BF16':173,'Q8_0':36}:
        raise RuntimeError('published GGUF tensor count/types changed')
    with safe_open(str(ASSETS/'model.safetensors'),framework='pt') as source:
        expected_names=set(source.keys())
        original_types=Counter(source.get_slice(name).get_dtype() for name in source.keys())
        source_shapes={name:tuple(source.get_slice(name).get_shape()) for name in source.keys()}
    gguf_names={tensor.name for tensor in reader.tensors}
    if gguf_names!=expected_names:
        raise RuntimeError(f'GGUF/checkpoint tensor names differ: missing={sorted(expected_names-gguf_names)[:4]} extra={sorted(gguf_names-expected_names)[:4]}')
    source_info={
        'source_gguf_sha256':observed_hash,
        'converter_version':CONVERTER_VERSION,
        'transformers_revision':PINS['transformers']['revision'],
        'tensor_count':len(reader.tensors),
        'tensor_types':tensor_types(reader),
        'model_safetensors_dtypes':dict(sorted(original_types.items())),
        'source_config_sha256':sha256(ASSETS/'config.json'),
        'output_sha256':sha256(path) if path.exists() else None,
        'config_sha256':sha256(OUT/'config.json') if (OUT/'config.json').exists() else None,
    }
    if path.exists() and marker.exists():
        previous=json.loads(marker.read_text())
        if all(previous.get(key)==value for key,value in source_info.items()) and previous.get('output_sha256')==sha256(path):
            return OUT
    state={}
    for tensor in reader.tensors:
        values=dequantize(tensor.data,tensor.tensor_type).reshape(tuple(reversed(tensor.shape))).copy()
        if values.shape!=source_shapes[tensor.name]:
            raise RuntimeError(f'GGUF shape mismatch for {tensor.name}: {values.shape} != {source_shapes[tensor.name]}')
        state[tensor.name]=torch.from_numpy(values)
    if set(state)!=expected_names or len(state)!=209:
        raise RuntimeError('not every published GGUF tensor was reconstructed')
    temp=tempfile.NamedTemporaryFile(prefix='model-',suffix='.safetensors.tmp',dir=OUT,delete=False)
    temp_path=Path(temp.name); temp.close()
    try:
        save_file(state,str(temp_path))
        os.replace(temp_path,path)
    finally:
        if temp_path.exists(): temp_path.unlink()
    shutil.copy2(ASSETS/'config.json',OUT/'config.json')
    source_info['config_sha256']=sha256(OUT/'config.json')
    source_info['output_sha256']=sha256(path)
    marker_tmp=OUT/'derived-manifest.json.tmp'
    marker_tmp.write_text(json.dumps(source_info,indent=2)+'\n')
    os.replace(marker_tmp,marker)
    summary={'source_model_safetensors':dict(sorted(original_types.items())),'published_gguf':tensor_types(reader),'q8_tensor_names':sorted(t.name for t in reader.tensors if t.tensor_type.name=='Q8_0'),'all_tensor_names_verified':len(gguf_names)==len(expected_names)}
    (Path(__file__).parent/'tensor-types.json').write_text(json.dumps(summary,indent=2)+'\n')
    return OUT

if __name__=='__main__': print(prepare_q8_checkpoint())
