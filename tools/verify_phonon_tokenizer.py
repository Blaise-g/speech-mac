import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tokenizers
from tokenizers import Tokenizer

root = Path(__file__).resolve().parents[1]
asset = Path(os.environ.get('PHONON_MODEL_DIR','/Volumes/Code/Personal/speech-mac-models/phonon-40m-en'))/'tokenizer.json'
raw = asset.read_bytes()
if hashlib.sha256(raw).hexdigest() != '691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936':
    raise SystemExit('tokenizer.json hash does not match pinned model revision')
doc = json.loads(raw)
if tokenizers.__version__ != '0.23.3':
    raise SystemExit(f'need tokenizers 0.23.3, got {tokenizers.__version__}')
oracle = Tokenizer.from_file(str(asset))
folder = root / '.build/phonon-tokenizer-check'
folder.mkdir(parents=True, exist_ok=True)
library = folder / 'tokenizer.dylib'
subprocess.run(['xcrun', 'clang', '-std=c17', '-O2', '-Wall', '-Wextra', '-Werror', '-Isrc', '-dynamiclib', 'src/phonon_tokenizer.c', '-o', str(library)], cwd=root, check=True)
native = C.CDLL(str(library))
libc = C.CDLL(None)
libc.free.argtypes = [C.c_void_p]

class Ids(C.Structure):
    _fields_ = [('ids', C.POINTER(C.c_uint32)), ('length', C.c_size_t)]

native.phonon_tokenizer_encode.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Ids)]
native.phonon_tokenizer_encode.restype = C.c_int
native.phonon_token_ids_free.argtypes = [C.POINTER(Ids)]
native.phonon_tokenizer_decode.argtypes = [C.POINTER(C.c_uint32), C.c_size_t, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
native.phonon_tokenizer_decode.restype = C.c_int

def encode(text):
    data = text.encode('utf-8')
    buffer = C.create_string_buffer(data)
    ids = Ids()
    if not native.phonon_tokenizer_encode(buffer, len(data), C.byref(ids)):
        raise AssertionError(f'native encoding failed: {text!r}')
    try:
        return list(ids.ids[:ids.length])
    finally:
        native.phonon_token_ids_free(C.byref(ids))

def decode(ids):
    values = (C.c_uint32 * len(ids))(*ids)
    result = C.c_void_p()
    size = C.c_size_t()
    if not native.phonon_tokenizer_decode(values, len(ids), C.byref(result), C.byref(size)):
        raise AssertionError(f'native decoding failed: {ids!r}')
    try:
        return C.string_at(result, size.value)
    finally:
        libc.free(result)

rng = random.Random(20261025)
pieces = [piece for piece, _ in doc['model']['vocab'] if '\0' not in piece]
texts = pieces + ['', ' ', '  ', '▁', '▁▁hello', 'a.a.a', '<s>', '<0xFF>', 'a\tb\nc', 'éclair', 'ß', '\U00010d70']
texts += [rng.choice(pieces) + rng.choice(['', ' ', '  ', '\t', '▁']) + rng.choice(pieces) for _ in range(2000)]
def scalar():
    while True:
        cp = rng.randrange(1, 0x110000)
        if not 0xd800 <= cp <= 0xdfff:
            return chr(cp)
texts += [''.join(scalar() for _ in range(rng.randrange(1, 8))) for _ in range(500)]
errors = []
encode_checked = 0
for text in texts:
    encode_checked += 1
    expected = oracle.encode(text, add_special_tokens=False).ids
    actual = encode(text)
    if actual != expected:
        errors.append({'kind': 'encode', 'input': repr(text), 'expected': expected, 'actual': actual})
        if len(errors) >= 12:
            break

byte_ids = {int(p[3:5], 16): i for i, (p, _) in enumerate(doc['model']['vocab']) if len(p) == 6 and p.startswith('<0x') and p.endswith('>')}
specials = [v['id'] for v in doc['added_tokens'] if v['special']]
sequences = [[i] for i in range(4000)]
sequences += [[byte_ids[0xe2], 0, byte_ids[0x96], byte_ids[0x81]], [byte_ids[0xc3], 203, byte_ids[0xa9]], [], [0, 1, 2, 203]]
sequences += [[rng.randrange(4000) for _ in range(rng.randrange(2, 9))] for _ in range(3000)]
sequences += [[rng.choice(list(byte_ids.values()) + specials) for _ in range(rng.randrange(2, 9))] for _ in range(2000)]
decode_checked = 0
for sequence in sequences:
    decode_checked += 1
    expected = oracle.decode(sequence, skip_special_tokens=True).encode('utf-8')
    actual = decode(sequence)
    if actual != expected:
        errors.append({'kind': 'decode', 'input': sequence, 'expected': repr(expected), 'actual': repr(actual)})
        if len(errors) >= 24:
            break
print(json.dumps({'library': str(library), 'encode_total': len(texts), 'encode_checked': encode_checked, 'decode_total': len(sequences), 'decode_checked': decode_checked, 'mismatches': errors}, ensure_ascii=True, indent=2))
raise SystemExit(bool(errors))
