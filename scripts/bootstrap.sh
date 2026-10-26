#!/bin/sh
# Check the host before uv creates an environment or contacts a registry.
set -eu
fail() { printf '%s\n' "bootstrap: $*" >&2; exit 1; }
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail 'requires Apple Silicon macOS (Darwin/arm64).'
command -v uv >/dev/null 2>&1 || fail 'install uv from https://docs.astral.sh/uv/getting-started/installation/ then retry.'
command -v make >/dev/null 2>&1 || fail 'install Xcode Command Line Tools with xcode-select --install, then retry.'
command -v xcrun >/dev/null 2>&1 && xcrun --find clang >/dev/null 2>&1 || fail 'install/select Xcode Command Line Tools with xcode-select --install, then retry.'
cd "$(dirname "$0")/.."
export PHONON_MODEL_DIR="${PHONON_MODEL_DIR:-$HOME/speech-mac-models/phonon-40m-en}"
uv sync --project reference/phonon --locked
uv run --project reference/phonon --no-sync --locked python scripts/bootstrap.py
make build
printf '%s\n' 'Ready. Save a WAV: .build/speech-mac say "Hello." --output hello.wav' 'Play it yourself: afplay hello.wav'
