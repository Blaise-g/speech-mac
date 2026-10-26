#!/bin/sh
# Run at repository root after the asset/fixture setup in report.md.
set -eu
: "${PHONON_MODEL_DIR:?Set the absolute shared pinned asset directory}"
case "$PHONON_MODEL_DIR" in /*) ;; *) echo 'PHONON_MODEL_DIR must be absolute' >&2; exit 2;; esac
export PHONON_MODEL_DIR UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache
export PHONON_DEFAULT_VOICE="$PHONON_MODEL_DIR/default-voice.safetensors"
export PHONON_NATIVE_PACK="$PWD/.build/phonon-native-f6acc8f43cb296be-v1.kpack"
export PHONON_CLI_PACK="$PHONON_NATIVE_PACK"
export PHONON_PROJECTED_VOICE="$PWD/.build/phonon-cli-projected.safetensors"
export PHONON_ENGINE_FIXTURE="$PWD/.build/phonon-engine-v1.bin"
export PHONON_PCM_FIXTURE="$PWD/.build/phonon-pcm-q8-v1.bin"
export PHONON_MIMI_FIXTURE="$PWD/.build/phonon-mimi-q8-v1.bin"
export PHONON_CONV_FIXTURE="$PWD/.build/phonon-conv-q8-v1.bin"
export PHONON_Q8_ORACLE="$PWD/.build/phonon-native-q8-projections-v1.bin"
export PHONON_BF16_ORACLE="$PWD/.build/phonon-native-bf16-oracle-v1.bin"
export PHONON_FLOW_SAMPLER_FIXTURE="$PWD/.build/phonon-flow-sampler-v1.bin"
export PHONON_FLOW_FIXTURE="$PWD/.build/phonon-flow-prefix-v1.bin"
export PHONON_CONDITION_FIXTURE="$PWD/.build/phonon-flow-condition-v1.bin"
unset PHONON_ENGINE_RECORD PHONON_STREAM_DEVICE
for fixture in "$PHONON_DEFAULT_VOICE" "$PHONON_NATIVE_PACK" "$PHONON_PROJECTED_VOICE" \
    "$PHONON_ENGINE_FIXTURE" "$PHONON_PCM_FIXTURE" "$PHONON_MIMI_FIXTURE" \
    "$PHONON_CONV_FIXTURE" "$PHONON_Q8_ORACLE" "$PHONON_BF16_ORACLE" \
    "$PHONON_FLOW_SAMPLER_FIXTURE" "$PHONON_FLOW_FIXTURE" "$PHONON_CONDITION_FIXTURE"; do
    test -s "$fixture" || { echo "Missing required acceptance fixture: $fixture" >&2; exit 2; }
done
make format
make check TEST_CFLAGS=-O2
make phonon-reference-check
make phonon-native-check TEST_CFLAGS=-O2
