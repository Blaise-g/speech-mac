import os,pathlib,subprocess,tempfile
root=pathlib.Path.cwd(); fresh=pathlib.Path(tempfile.mkdtemp(prefix='speech-mac-34-fresh-online-')); cache=tempfile.mkdtemp(prefix='speech-mac-34-uv-cache-')
archive=subprocess.Popen(['git','archive','4753207c6f5e3c188f665686a5c9c9f1b22d88ac'],stdout=subprocess.PIPE)
subprocess.run(['tar','-x','-C',str(fresh)],stdin=archive.stdout,check=True); assert archive.wait()==0
print('FRESH_TRACKED_SOURCE=4753207c6f5e3c188f665686a5c9c9f1b22d88ac',flush=True); print('FRESH_DIRECTORY='+str(fresh),flush=True); print('ISOLATED_UV_CACHE='+cache,flush=True)
env=dict(os.environ,UV_CACHE_DIR=cache);env.pop('VIRTUAL_ENV',None)
subprocess.run(['sh','-c',(root/'artifacts/34-acceptance-20261026/fresh-setup.sh').read_text()],cwd=fresh,env=env,check=True)
subprocess.run(['git','diff','--exit-code','4753207c6f5e3c188f665686a5c9c9f1b22d88ac','--','reference/phonon/uv.lock'],cwd=root,check=True)
print('FRESH_BUILD_SETUP_EXIT=0',flush=True)
(root/'.build/ticket34/fresh-directory.txt').write_text(str(fresh)+'\n')
