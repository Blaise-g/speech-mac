"""Download and verify the immutable official model artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODEL_DIR / ".huggingface"))
HF_REPOSITORY = "hexgrad/Kokoro-82M"
HF_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
ARTIFACTS = {
    "config.json": ("14a726edd3718279eac426630879ff743955b16a", "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f", 2351),
    "kokoro-v1_0.pth": ("496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4", "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4", 327212226),
    "voices/af_heart.pt": ("0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff", "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff", 523425),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets() -> dict[str, str]:
    """Require the exact pinned config, checkpoint, and af_heart style files."""
    actual: dict[str, str] = {}
    failures: list[str] = []
    for filename, (_, expected_sha, expected_size) in ARTIFACTS.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            failures.append(f"missing {path.relative_to(ROOT)}")
            continue
        actual_sha = _sha256(path)
        actual[filename] = actual_sha
        if path.stat().st_size != expected_size:
            failures.append(f"wrong size for {filename}: {path.stat().st_size}, expected {expected_size}")
        if actual_sha != expected_sha:
            failures.append(f"wrong SHA-256 for {filename}: {actual_sha}, expected {expected_sha}")
    if failures:
        details = "; ".join(failures)
        raise RuntimeError(
            f"model artifacts are missing or do not match pins: {details}. "
            "Run `uv run --project reference --locked python reference/assets.py download`."
        )
    return actual


def download_assets() -> None:
    """Fetch the pinned public files into the ignored models/ directory."""
    os.environ.setdefault("HF_HOME", str(MODEL_DIR / ".huggingface"))
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACTS:
        hf_hub_download(
            repo_id=HF_REPOSITORY,
            filename=filename,
            revision=HF_REVISION,
            local_dir=str(MODEL_DIR),
        )
    verify_assets()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "verify"))
    arguments = parser.parse_args()
    if arguments.command == "download":
        download_assets()
    else:
        verify_assets()
    print("Pinned model config, checkpoint, and af_heart voice verified.")
