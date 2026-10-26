"""Verify shared model assets against checked-in revision-specific digest pins."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(os.environ.get('PHONON_MODEL_DIR','models/phonon-40m-en'))
PINS=Path(__file__).with_name('pins.json')

def verified_pins(root=None):
    root = ROOT if root is None else Path(root)
    pins=json.loads(PINS.read_text())
    if pins['model']['revision']!='cc738b7e4d182c2ba1bafb8062aa4846af000637' or pins['mimi']['revision']!='33519be1a8e61c6fa00c6e154e2d38e76a367919': raise SystemExit('model revision pin drift')
    for record in pins['files']:
        path=root/record['path']
        try:
            data=path.read_bytes()
        except OSError as error:
            raise SystemExit(f'asset unavailable: {path}: {error}') from error
        if len(data)!=record['size_bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise SystemExit(f'asset hash/size mismatch: {record["repo"]}@{record["revision"]}:{record["path"]}')
    return pins

def verify():
    return len(verified_pins()['files'])

if __name__=='__main__': print(f'verified {verify()} assets against committed pins')
