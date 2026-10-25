import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

root = Path(__file__).resolve().parents[1]
folder = root / '.build/phonon-prompt-prepare-oracle'
folder.mkdir(parents=True, exist_ok=True)
source_path = root / '.build/xn-ptts/repo/ptts/src/tts_model.rs'
source_bytes = source_path.read_bytes()
source_sha256 = 'a9a57097d5786570a48948c02a91840409e2725bfb7887ce00b3d65e2ceffbe5'
if hashlib.sha256(source_bytes).hexdigest() != source_sha256:
    raise SystemExit('pinned tts_model.rs hash mismatch')
source = source_bytes.decode()
prepare = 'pub fn prepare_text_prompt(' + source.split('pub fn prepare_text_prompt(', 1)[1].split('\n#[cfg(test)]', 1)[0]
main = r'''
use std::io::{self, BufRead};
fn main() {
 for line in io::stdin().lock().lines() {
  let line = line.unwrap();
  let bytes: Vec<u8> = line.as_bytes().chunks(2).map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(),16).unwrap()).collect();
  let (text, tail) = prepare_text_prompt(std::str::from_utf8(&bytes).unwrap());
  for b in text.as_bytes() { print!("{:02x}", b); }
  println!(":{}", tail);
 }
}
'''
rust = folder / 'prepare.rs'
rust.write_text(prepare + '\n' + main)
env = os.environ.copy()
env['CARGO_HOME'] = str(root / '.build/reference-toolchain/cargo')
env['RUSTUP_HOME'] = str(root / '.build/reference-toolchain/rustup')
binary = folder / 'prepare'
subprocess.run([str(root / '.build/reference-toolchain/cargo/bin/rustc'), str(rust), '-O', '-o', str(binary)], env=env, check=True)
library = folder / 'prompt.dylib'
subprocess.run(['xcrun', 'clang', '-std=c17', '-O2', '-Wall', '-Wextra', '-Werror', '-Isrc', '-dynamiclib', 'src/phonon_prompt_plan.c', 'src/phonon_tokenizer.c', '-o', str(library)], cwd=root, check=True)
native = C.CDLL(str(library))
libc = C.CDLL(None)
libc.free.argtypes = [C.c_void_p]
native.phonon_prepare_text_prompt.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
native.phonon_prepare_text_prompt.restype = C.c_int
texts = ['', ' ', 'hello WORLD', 'one two three four five', 'éclair', 'ßword', 'ﬃ text', 'ǰ', 'ΐ', '\U00010d70', 'a\u2003b', '\r\n\t', 'a.a.a', '🙂', 'end▁']
texts += [chr(cp) for cp in range(1, 0x110000, 137) if not 0xd800 <= cp <= 0xdfff]
rng = random.Random(20261025)
def scalar():
    while True:
        cp = rng.randrange(1, 0x110000)
        if not 0xd800 <= cp <= 0xdfff:
            return chr(cp)
texts += [scalar() + ' one\t two ' + scalar() for _ in range(1000)]
wire = ''.join(text.encode().hex() + '\n' for text in texts).encode()
expected = subprocess.run([str(binary)], input=wire, stdout=subprocess.PIPE, check=True).stdout.decode().splitlines()
if len(expected) != len(texts):
    raise SystemExit(f'Rust oracle returned {len(expected)} rows for {len(texts)} inputs')
errors = []
checked = 0
for text, row in zip(texts, expected):
    checked += 1
    hex_value, tail_value = row.rsplit(':', 1)
    want = bytes.fromhex(hex_value)
    data = text.encode()
    buf = C.create_string_buffer(data)
    out, length, tail = C.c_void_p(), C.c_size_t(), C.c_size_t()
    status = native.phonon_prepare_text_prompt(buf, len(data), C.byref(out), C.byref(length), C.byref(tail))
    actual = C.string_at(out, length.value) if out else None
    libc.free(out)
    if not status or actual != want or tail.value != int(tail_value):
        errors.append({'input': repr(text), 'expected': repr(want), 'expected_tail': tail_value, 'actual': repr(actual), 'tail': tail.value, 'status': status})
        if len(errors) >= 12:
            break
print(json.dumps({'checked': checked, 'total': len(texts), 'source_sha256': source_sha256, 'reference_function_sha256': hashlib.sha256(prepare.encode()).hexdigest(), 'mismatches': errors}, indent=2))
raise SystemExit(bool(errors))
