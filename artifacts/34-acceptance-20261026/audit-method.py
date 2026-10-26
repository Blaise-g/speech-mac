import hashlib,json,math,os,pathlib,subprocess,wave
root=pathlib.Path.cwd(); out=root/'artifacts/34-acceptance-20261026'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(args,**kw): return subprocess.check_output(args,text=True,**kw)
queue=(out/'queue-current.log').read_text(); r=next(json.loads(x) for x in queue.splitlines() if x.startswith('{'))
assert 'QUEUE_TIMING_EXIT=0' in queue and 'EXEC_COMMAND_EXIT=0' in queue
assert r['consumer']=='queue_volume_zero_AudioQueue' and r['acoustic_onset_s'] is None
assert r['samples']==76800 and r['frames']==r['reclaimed_buffers']==40
keys=('first_pcm_s','first_submission_s','running_observed_s','first_reclaimed_s','synthesis_end_s','drain_s')
assert all(math.isfinite(r[k]) and r[k]>0 for k in keys)
assert r['first_pcm_s']<=r['first_submission_s']<=r['running_observed_s']<r['synthesis_end_s']<=r['drain_s']
assert r['first_submission_s']<=r['first_reclaimed_s']<=r['drain_s']
fresh=pathlib.Path((root/'.build/ticket34/fresh-directory.txt').read_text().strip())
binaries=[root/'.build/speech-mac',fresh/'.build/speech-mac']
audit={'source_commit':run(['git','rev-parse','HEAD']).strip(),'runtime_source_commit':'cef25447a45840db848f6e3aadf0f0d9afc8272c','host':run(['uname','-a']).strip(),'compiler':run(['xcrun','clang','--version']).strip(),'queue_binary_sha256':sha(root/'.build/34-current-queue'),'queue':r,'binaries':[],'auditions':[]}
for binary in binaries:
    libs=run(['otool','-L',str(binary)]); symbols=run(['nm','-g',str(binary)])
    assert all(x not in libs.lower() for x in ['python','rust','espeak','coreml','onnx','torch'])
    assert all(x not in symbols for x in ['kokoro_','speech_test_synthesize','speech_profile_begin','phonon_pack_prepare_linears_for_tests'])
    assert run([str(binary),'--version'])=='speech-mac 0.1.0\n'
    audit['binaries'].append({'path':str(binary),'sha256':sha(binary),'libraries':libs,'undefined_symbols':run(['nm','-u',str(binary)])})
# The source census is a call-site audit, not an OS-level child-process trace.
spawn=[]
for p in sorted((root/'src').glob('*.c')):
    for n,line in enumerate(p.read_text().splitlines(),1):
        if any(token in line for token in ('posix_spawn(', 'fork(', 'execve(', 'popen(', 'system(')):
            spawn.append({'path':str(p.relative_to(root)),'line':n,'source':line.strip()})
assert len(spawn)==1 and spawn[0]['path']=='src/speech_cli.c'
audit['process_creation_call_sites']=spawn
assert sha(fresh/'.build/fresh-default.wav')==sha(root/'.build/cli-pos.wav')
audit['fresh_default_matches_cli_seed123']=True
for name in ['short','structured','long']:
    current=root/'.build/ticket34'/f'{name}.wav'; historical=root/'artifacts/32-phonon-baseline-20261026'/f'{name}.wav'
    assert current.read_bytes()==historical.read_bytes()
    with wave.open(str(current)) as w:
        assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,24000)
        count=w.getnframes()
    audit['auditions'].append({'workload':name,'current_generated_path':str(current),'reused_artifact':str(historical.relative_to(root)),'sha256':sha(current),'samples':count,'seed':0,'duration_delta':0,'language':'en','text_path':f'artifacts/32-phonon-baseline-20261026/{name}.txt','text_sha256':sha(root/f'artifacts/32-phonon-baseline-20261026/{name}.txt'),'listening':'PENDING user confirmation'})
for p in sorted(out.iterdir()):
    if p.suffix=='.log':
        assert p.stat().st_size
print('PASS current queue strict events/counts; fresh/current native binary dependencies/symbols/call sites; fresh default seed123 identity; current audition files identical to tracked32 WAVs')
# Record immutable input identities and exact source of every compiled runtime unit.
audit['pack_sha256']=sha(root/'.build/phonon-native-f6acc8f43cb296be-v1.kpack')
audit['voice_sha256']=sha(pathlib.Path('/Volumes/Code/Personal/speech-mac-models/phonon-40m-en/default-voice.safetensors'))
audit['pins_sha256']=sha(root/'reference/phonon/pins.json')
audit['lock_sha256']=sha(root/'reference/phonon/uv.lock')
audit['runtime_source_sha256']={str(p.relative_to(root)):sha(p) for p in sorted((root/'src').rglob('*')) if p.is_file()}
for p in (root/'src').rglob('*'):
    if p.is_file():
        assert hashlib.sha256(subprocess.check_output(['git','show','cef25447a45840db848f6e3aadf0f0d9afc8272c:'+str(p.relative_to(root))])).hexdigest()==sha(p)
assert sha(fresh/'reference/phonon/uv.lock')==audit['lock_sha256']
assert sha(fresh/'models/phonon-40m-en.kpack')==audit['pack_sha256']
assert sha(fresh/'voices/default.safetensors')==audit['voice_sha256']
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('PASS immutable runtime source matches optimized sourcecef2544; fresh lock/native pack/default voice match verified inputs')
