#!/usr/bin/env python3
"""Generate the pinned native Unigram table from verified Phonon tokenizer.json."""
import hashlib, json, pathlib, sys
import os

MODEL_REV = "cc738b7e4d182c2ba1bafb8062aa4846af000637"
EXPECTED_SHA256 = "691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936"
ROOT=pathlib.Path(__file__).resolve().parents[1]
ASSETS=pathlib.Path(os.environ.get("PHONON_MODEL_DIR","/Volumes/Code/Personal/speech-mac-models/phonon-40m-en"))
source = pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else ASSETS/"tokenizer.json"
out = pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/"src/phonon_tokenizer_vocab.inc"
data = source.read_bytes()
def require(condition,message):
 if not condition: raise SystemExit(message)
require(hashlib.sha256(data).hexdigest()==EXPECTED_SHA256,"tokenizer.json hash does not match pinned model revision")
doc=json.loads(data)
m=doc["model"]
require(doc["version"]=="1.0" and doc["truncation"] is None and doc["padding"] is None and doc["post_processor"] is None,"tokenizer preprocessing metadata differs from the pinned contract")
require(m["type"]=="Unigram" and len(m["vocab"])==4000 and m["unk_id"]==0 and m["byte_fallback"] is True,"tokenizer model metadata differs from the pinned contract")
require(all(isinstance(piece,str) and isinstance(score,(int,float)) for piece,score in m["vocab"]),"invalid vocabulary piece or score")
added=doc["added_tokens"]
require(len(added)==204 and [x["id"] for x in added]==list(range(204)),"added-token ID set differs from the pinned metadata")
require(all(x["special"] and x["content"].startswith("\x00") and not x["single_word"] and not x["lstrip"] and not x["rstrip"] and not x["normalized"] for x in added),"added-token special metadata differs from the pinned metadata")
require(doc["normalizer"]=={"type":"Prepend","prepend":"▁"},"normalizer metadata differs from the pinned contract")
require(doc["pre_tokenizer"]=={"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":True},"pre-tokenizer metadata differs from the pinned contract")
decoder=doc["decoder"]
require(decoder["type"]=="Sequence" and decoder["decoders"]==[
 {"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":True},
 {"type":"ByteFallback"},{"type":"Fuse"}],"decoder metadata differs from the pinned contract")
with out.open("w") as f:
 f.write("/* Generated from gradium/phonon-40m-en tokenizer.json @ "+MODEL_REV+"; sha256 "+EXPECTED_SHA256+". */\n")
 f.write("static const char *const phonon_vocab[4000] = {\n")
 for piece,score in m["vocab"]: f.write("  "+json.dumps(piece[1:] if piece.startswith("\x00") else piece,ensure_ascii=True)+",\n")
 f.write("};\nstatic const double phonon_scores[4000] = {\n")
 for _,score in m["vocab"]: f.write("  "+repr(float(score))+",\n")
 f.write("};\n")
 pieces=[(piece[1:] if piece.startswith("\x00") else piece, i) for i,(piece,_) in enumerate(m["vocab"])]
 buckets=[[] for _ in range(256)]
 for piece,i in pieces:
  if i>=204 and piece: buckets[piece.encode("utf-8")[0]].append((len(piece.encode("utf-8")),i))
 for bucket in buckets: bucket.sort()
 f.write(f"#define PHONON_VOCAB_MAX_BYTES {max(len(piece.encode('utf-8')) for piece,_ in pieces)}U\n")
 f.write("static const unsigned short phonon_candidate_offsets[257] = {\n")
 offset=0
 for bucket in buckets: f.write(f" {offset},\n"); offset+=len(bucket)
 f.write(f" {offset}\n}};\nstatic const unsigned short phonon_candidate_ids[{offset}] = {{\n")
 f.write(" "+", ".join(str(i) for bucket in buckets for _,i in bucket)+"\n};\n")
 f.write("static const unsigned short phonon_byte_id[256] = {\n")
 ids={int(p[3:5],16):i for i,(p,_) in enumerate(m["vocab"]) if len(p)==6 and p.startswith("<0x") and p.endswith(">")}
 if len(ids)!=256: raise SystemExit(f"expected 256 byte tokens, got {len(ids)}")
 for i in range(256): f.write(f" {ids[i]}{',' if i<255 else ''}\n")
 reverse={token_id:byte for byte,token_id in ids.items()}
 f.write("};\nstatic const unsigned short phonon_byte_value[4000] = {\n")
 for i in range(4000): f.write(f" {reverse.get(i,65535)}{',' if i<3999 else ''}\n")
 f.write("};\n")
