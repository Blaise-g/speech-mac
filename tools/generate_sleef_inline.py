"""Reproduce the pinned native math header; development only, never a build step.

UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference/phonon \
  --offline --no-sync --locked python tools/generate_sleef_inline.py [--archive PATH]
"""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "56e1f79cb140fb9326d612d0be06b5250565cade"
URL = f"https://codeload.github.com/shibatch/sleef/tar.gz/{REVISION}"
ARCHIVE_SHA256 = "d51c477ae25bead879e8fc869fb3740e66c18b431f45ed0598ec4ba4985023a0"
HEADER_SHA256 = "9c67f846daa49394628b4390396cb7223197e383de940e02cc3a58d9f33b477f"
LICENSE_SHA256 = "c9bff75738922193e67fa726fa225535870d2aa1059f91452c411736284ad566"
OPTIONS = ["-DSLEEF_BUILD_DFT=OFF", "-DSLEEF_BUILD_TESTS=OFF", "-DSLEEF_BUILD_GNUABI_LIBS=OFF",
           "-DSLEEF_BUILD_QUAD=OFF", "-DSLEEF_BUILD_INLINE_HEADERS=ON"]


def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"hash mismatch for {path.name}: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="verified local archive; avoids network")
    parser.add_argument("--output", type=Path, default=ROOT / "vendor/sleef/sleefinline_advsimd.h")
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="phonon-sleef-generate-"))
    archive = args.archive
    if archive is None:
        archive = scratch / "sleef.tar.gz"
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    verify(archive, ARCHIVE_SHA256)
    with tarfile.open(archive) as source:
        source.extractall(scratch, filter="data")
    source = scratch / f"sleef-{REVISION}"
    verify(source / "LICENSE.txt", LICENSE_SHA256)
    build = scratch / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), *OPTIONS], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "inline_headers_util", "-j", "4"], check=True)
    header = build / "include/sleefinline_advsimd.h"
    verify(header, HEADER_SHA256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_bytes() != header.read_bytes():
        shutil.copyfile(header, args.output)
    print(f"Reproduced {header.stat().st_size} bytes; sha256={HEADER_SHA256}")
    print(f"Output: {args.output}; development scratch: {scratch}")


if __name__ == "__main__":
    main()
