#!/bin/sh
# Run inside a clean tracked-source checkout. Choose a persistent asset directory.
set -eu
: "${PHONON_MODEL_DIR:?Set an absolute persistent directory outside disposable checkouts}"
case "$PHONON_MODEL_DIR" in /*) ;; *) exit 2;; esac
export PHONON_MODEL_DIR
export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/speech-mac-uv-cache}"
uv sync --project reference/phonon --locked
# First-time gated setup: accept the model terms and authenticate locally, then:
# uv run --project reference/phonon --locked python reference/phonon/download_assets.py
# Existing pinned assets may be reused read-only; never substitute another model.
uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/verify_assets.py
make phonon-native-pack
mkdir -p models voices
# Copies prevent mutation checks in this checkout from modifying shared assets.
cp .build/phonon-native-f6acc8f43cb296be-v1.kpack models/phonon-40m-en.kpack
cp "$PHONON_MODEL_DIR/default-voice.safetensors" voices/default.safetensors
make build
# Explicit output saves audio without playback. No streaming/device action here.
PATH=/usr/bin:/bin .build/speech-mac say 'Hello.' --seed 123 --output .build/fresh-default.wav
.build/speech-mac voices
