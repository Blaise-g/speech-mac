#!/usr/bin/env python3
"""Compile source-matched prompt planner vectors from tokenizers 0.23.3.

Policy is ported from pinned xn-ptts ptts/src/tts_model.rs and plan.rs at
ac0e1524cfd9e71d57aadca0de542796f942d775. Prompt preparation cases here use
ASCII first scalars; Unicode 17 preparation is checked against Rust separately.
C runtime is not imported here.
"""
import hashlib
import json
import math
import os
import pathlib
import sys
import tokenizers
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[1]
ASSETS=pathlib.Path(os.environ.get('PHONON_MODEL_DIR','/Volumes/Code/Personal/speech-mac-models/phonon-40m-en'))
ASSET=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else ASSETS/'tokenizer.json'
OUT=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'tests/phonon_prompt_plan_vectors.inc'
TOKENIZER_SHA='691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936'
SOURCE_ROOT=ROOT/'.build/xn-ptts/repo'
SOURCE_HASHES=json.loads((ROOT/'tools/prompt_plan_source_hashes.json').read_text())['files']
CASES=[
 ('Hello. Hi.',3,0),
 ('One. Two! Three? Four...',1,0),
 ('a.a.a',50,0),
 ('one two three four five, six seven eight nine ten eleven twelve',50,8),
 ('red, one two three four five six seven eight nine ten eleven twelve',50,9),
 ('one two three four "five," six seven eight nine ten eleven twelve',50,8),
 ('one two three four five,} six seven eight nine ten eleven twelve',50,8),
 ('one two three four “five,” six seven eight nine ten eleven twelve',50,8),
 ('▁hello. world!',50,0),
 ('one two three four five six seven eight',50,8),
]
if tokenizers.__version__!='0.23.3':raise SystemExit(f'need tokenizers 0.23.3, got {tokenizers.__version__}')
if hashlib.sha256(ASSET.read_bytes()).hexdigest()!=TOKENIZER_SHA:raise SystemExit('tokenizer asset hash mismatch')
for name,expected in SOURCE_HASHES.items():
 path=SOURCE_ROOT/name
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
  raise SystemExit(f'pinned planner source missing or changed: {path}')
tok=Tokenizer.from_file(str(ASSET))
def enc(s):return tok.encode(s,add_special_tokens=False).ids
def prep(text):
 text=' '.join(text.strip().replace('\n',' ').replace('\r',' ').split())
 words=text.split();tail=3 if len(words)<=4 else 1
 if text:text=text[0].upper()+text[1:]
 if text and text[-1].isalnum():text+='.'
 return text,tail
def split(text,max_tokens):
 prepared,_=prep(text);prepared=prepared.strip();ids=enc(prepared);marks=enc('.!...?');eos=marks[1:] if len(marks)>1 else marks
 bounds=[0];prev=False
 for i,token in enumerate(ids):
  if token in eos:prev=True
  else:
   if prev:bounds.append(i)
   prev=False
 bounds.append(len(ids));sentences=[]
 for a,b in zip(bounds,bounds[1:]):
  if b>a:sentences.append((b-a,tok.decode(ids[a:b],skip_special_tokens=True)))
 chunks=[];group='';count=0
 for n,sentence in sentences:
  if not group:group=sentence;count=n;continue
  if count+n>max_tokens:chunks.append(group.strip());group=sentence;count=n
  else:group+=' '+sentence;count+=n
 if group:chunks.append(group.strip())
 return chunks
def chunk(text,rate=12.5):
 prepared,tail=prep(text);ids=enc(prepared);frames=math.ceil((len(ids)/3+2)*rate)
 return {'text':text,'ids':ids,'tail':tail,'frames':frames,'seq':len(ids)+512+frames}
def cut(words,min_side):
 ends=[];total=0
 for word in words:total+=len(enc(word));ends.append(total)
 inner=list(enumerate(ends[:-1]));imbalance=lambda item:abs(item[1]-(total-item[1]));closers='"\') ]”’»'.replace(' ','')
 def after(marks):
  return min((item for item in inner if words[item[0]].rstrip(closers).endswith(tuple(marks)) and item[1]>=min_side and total-item[1]>=min_side),key=imbalance,default=None)
 found=after('.!?')
 if found is None:found=after(',;:')
 if found is None:found=min(inner,key=imbalance,default=None)
 if found is None:raise ValueError('nothing to cut')
 return found[0]+1
def fit(chunks,max_tokens):
 todo=list(reversed(chunks));out=[]
 while todo:
  current=todo.pop()
  if len(current['ids'])<=max_tokens:out.append(current);continue
  words=current['text'].split()
  if len(words)>=2:
   at=cut(words,(max_tokens+3)//4);halves=[' '.join(words[:at]),' '.join(words[at:])]
  else:
   word=current['text'].strip()
   if len(current['ids'])<=max_tokens:out.append(current);continue
   chars=list(word);mid=len(chars)//2
   if mid==0:raise ValueError('unsplittable scalar')
   halves=[''.join(chars[:mid]),''.join(chars[mid:])]
  for half in reversed(halves):todo.append(chunk(half))
 return out
def cbytes(s):return '"'+''.join('\\x%02x""'%b for b in s.encode())+'"'
with OUT.open('w') as f:
 f.write('/* Generated from HF tokenizers 0.23.3 and xn-ptts ac0e1524; source hashes: tools/prompt_plan_source_hashes.json. */\n')
 vector_rows=[]
 for i,(text,group,fit_max) in enumerate(CASES):
  chunks=[chunk(s) for s in split(text,group)]
  if fit_max:chunks=fit(chunks,fit_max)
  for j,c in enumerate(chunks):
   f.write(f'static const uint32_t prompt_ids_{i}_{j}[] = {{{", ".join(map(str,c["ids"])) or "0"}}};\n')
   f.write(f'static const unsigned char prompt_text_{i}_{j}[] = {cbytes(c["text"])};\n')
  f.write(f'static const struct PromptExpected prompt_expected_{i}[] = {{\n')
  for j,c in enumerate(chunks):f.write(f' {{{j},prompt_text_{i}_{j},sizeof(prompt_text_{i}_{j})-1,prompt_ids_{i}_{j},{len(c["ids"])},{c["tail"]},{c["frames"]},{c["seq"]}}},\n')
  f.write('};\n')
  vector_rows.append(f' {{{cbytes(text)},{len(text.encode())},{group},{fit_max},prompt_expected_{i},{len(chunks)}}},\n')
 f.write('static const struct PromptVector prompt_vectors[] = {\n')
 f.writelines(vector_rows)
 f.write('};\n')
