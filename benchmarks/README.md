# Benchmarks

Current Phonon measurements are the [native baseline](../artifacts/32-phonon-baseline-20261026/report.md) and [linear-weight preparation comparison](../artifacts/33-linear-materialization-20261026/report.md). Earlier model reports and their removed runners remain in [Git history](https://github.com/Blaise-g/speech-mac/tree/3434f73aeada7e1818feb11cf4815b773a21188b/artifacts); they establish no Phonon speed claim.

Run the native Phonon baseline from a clean committed checkout with verified local assets:

```sh
export PHONON_MODEL_DIR="$HOME/speech-mac-models/phonon-40m-en"
uv run --project reference/phonon --offline --no-sync --locked \
  python reference/phonon/benchmark.py \
  --pack .build/phonon-native-f6acc8f43cb296be-v1.kpack \
  --voice "$PHONON_MODEL_DIR/default-voice.safetensors" \
  --output .build/phonon-baseline-new
```

The driver verifies source assets against committed `reference/phonon/pins.json` before creating output; no machine-local verification manifest is required. Follow the [asset setup](../reference/phonon/README.md#asset-setup) first. The output directory must be new. Five repetitions per workload/mode are the default; supported counts are 5–19. Reserve the host for one measurement writer and do not run tests or other benchmarks concurrently. The driver writes every measured repetition to `raw.json`, retained excluded warmups, a summary report and unplayed audition WAVs. It builds separate release and profile binaries, verifies source-asset hashes, and rejects differing F32 PCM across repetitions and profile/release builds. Fresh process does not mean a cold OS cache.

`phonon_queue.c` is a separate bounded real-device probe. Build all engine sources with `-DSPEECH_STREAM_SILENT_OUTPUT`; an ordinary build refuses device access. It changes only the new queue's volume, emits observed queue events, and cannot measure acoustic onset. Fake-consumer stream tests remain separate. Device/listening acceptance requires its own evidence; do not infer either from synthesis timings or numerical parity.
