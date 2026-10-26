"""Prepare verified local runtime assets; never replace a different user artifact."""
from __future__ import annotations

import filecmp
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "phonon"))
from verify_assets import verified_pins


def check_destination(source: Path, target: Path) -> bool:
    """Return whether installation is needed, rejecting existing custom data."""
    if target.exists() or target.is_symlink():
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            return False
        raise RuntimeError(f"{target} already exists with different contents; move it aside explicitly and retry")
    return True


def install(source: Path, target: Path) -> None:
    if not check_destination(source, target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
        # A hard link publishes complete bytes without overwriting a racing creator.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_assets(assets: Path) -> None:
    try:
        verified_pins(assets)
    except SystemExit:
        print("Pinned assets are missing or corrupt; downloading the canonical snapshots.", flush=True)
        try:
            from download_assets import download
            download(assets)
            verified_pins(assets)
        except (Exception, SystemExit) as error:
            raise RuntimeError(
                "could not obtain verified assets. Accept the gated terms at "
                "https://huggingface.co/gradium/phonon-40m-en and authenticate locally with "
                "`uvx --from huggingface-hub hf auth login`; check network access and "
                "PHONON_MODEL_DIR, then retry. Credentials are never required for valid local assets."
            ) from error


def prepare(root: Path, assets: Path) -> None:
    ensure_assets(assets)
    voice = assets / "default-voice.safetensors"
    voice_target = root / "voices" / "default.safetensors"
    # Fail before converting or installing anything if the user's voice conflicts.
    check_destination(voice, voice_target)
    from native_pack import create
    with tempfile.TemporaryDirectory(prefix="speech-mac-bootstrap-") as temporary:
        pack = create(Path(temporary) / "model.kpack")
        pack_target = root / "models" / "phonon-40m-en.kpack"
        check_destination(pack, pack_target)
        install(pack, pack_target)
        install(voice, voice_target)
    print("Verified native pack and default voice are installed.")


if __name__ == "__main__":
    try:
        prepare(ROOT, Path(os.environ["PHONON_MODEL_DIR"]))
    except (OSError, RuntimeError) as error:
        sys.exit(f"bootstrap: {error}")
