#!/usr/bin/env python3
"""Refresh deterministic tokenizer fixtures from the locked HF tokenizers oracle."""
import hashlib
import os
import pathlib
import sys
import tokenizers
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = pathlib.Path(os.environ.get('PHONON_MODEL_DIR', '/Volumes/Code/Personal/speech-mac-models/phonon-40m-en'))
ASSET = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ASSETS / 'tokenizer.json'
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'tests/phonon_tokenizer_vectors.inc'
EXPECTED_SHA = '691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936'
CASES = ['', ' ', '  two  spaces ', 'a\tb', 'one\ntwo', 'éclair café', '你好，世界！', 'مرحبا بالعالم', '🙂', '🫠', 'curly “quotes,” and—dashes…', '<unk>', '<s>', '</s>', '<pad>', '<|c000|>', '<0xFF>', 'literal <s> then <0x00>', 'a\u2003b\u00a0c', 'quoted “yes,” please!', '.!...?', 'First. Second! Third? Four...', '▁hello']

def cbytes(value: str) -> str:
    data=value.encode('utf-8')
    return '"' + ''.join('\\x%02x""' % b for b in data) + '"'

raw=ASSET.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA: raise SystemExit('pinned tokenizer.json hash mismatch')
if tokenizers.__version__ != '0.23.3': raise SystemExit(f'need tokenizers 0.23.3, got {tokenizers.__version__}')
tok=Tokenizer.from_file(str(ASSET))
with OUT.open('w') as f:
    f.write('/* Generated with tokenizers 0.23.3 from pinned tokenizer.json sha256 '+EXPECTED_SHA+'. */\n')
    for i,text in enumerate(CASES):
        enc=tok.encode(text,add_special_tokens=False)
        ids=', '.join(map(str,enc.ids)) or '0'
        f.write(f'static const uint32_t tokenizer_ids_{i}[] = {{{ids}}};\n')
        f.write(f'static const unsigned char tokenizer_text_{i}[] = {cbytes(text)};\n')
        f.write(f'static const unsigned char tokenizer_decoded_{i}[] = {cbytes(tok.decode(enc.ids,skip_special_tokens=True))};\n')
    f.write('static const struct TokenizerVector tokenizer_vectors[] = {\n')
    for i,text in enumerate(CASES):
        enc=tok.encode(text,add_special_tokens=False)
        f.write(f' {{{"tokenizer_text_%d"%i}, sizeof(tokenizer_text_{i})-1, tokenizer_ids_{i}, {len(enc.ids)}, tokenizer_decoded_{i}, sizeof(tokenizer_decoded_{i})-1}},\n')
    f.write('};\n')
