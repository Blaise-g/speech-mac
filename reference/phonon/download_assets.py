"""Download the exact development assets in pins.json, verifying before installation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import hf_hub_download

PINS = Path(__file__).with_name("pins.json")


def download(destination: Path) -> int:
    records = json.loads(PINS.read_text())["files"]
    destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        cached = Path(hf_hub_download(repo_id=record["repo"], revision=record["revision"],
                                      filename=Path(record["path"]).name))
        data = cached.read_bytes()
        if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f'asset hash/size mismatch: {record["path"]}')
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(data)
                temporary.flush()
                os.replace(temporary_path, target)
            finally:
                temporary_path.unlink(missing_ok=True)
    return len(records)


if __name__ == "__main__":
    root = Path(os.environ.get("PHONON_MODEL_DIR", "models/phonon-40m-en"))
    print(f"downloaded and verified {download(root)} pinned assets")
