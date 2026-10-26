import os,pathlib,subprocess,tempfile
root=pathlib.Path.cwd(); original=(root/'artifacts/34-acceptance-20261026/acceptance.log').read_text(); temp=pathlib.Path(tempfile.mkdtemp(prefix='speech-mac-34-validator-red-')); target=temp/'artifacts/34-acceptance-20261026';target.mkdir(parents=True)
method=root/'artifacts/34-acceptance-20261026/validate-suite.py'
for name,text in [('wrong-engine-count',original.replace('engine actual record=0 samples=30720','engine actual record=0 samples=30721',1)),('model-skip',original+'\nSKIP real model\n')]:
    (target/'acceptance.log').write_text(text)
    p=subprocess.run(['uv','run','--project',str(root/'reference/phonon'),'--offline','--no-sync','--locked','python',str(method)],cwd=temp,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert p.returncode !=0
    print(name+': correctly rejected exit='+str(p.returncode))
