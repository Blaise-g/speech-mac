# 09 audition artifacts

Both files are mono PCM16 WAV at 24 kHz, with 33,000 samples (1.375 seconds). They use the
prepared phonemes `həˈloʊ`, IDs `[0, 50, 83, 156, 54, 57, 135, 0]` including BOS/EOS, `af_heart`
style row 5, speed 1.0, and seed 104729.

The controlled file uses the captured reference excitation and source STFT so the assembled
conditioning and learned generator can be compared with the pinned final waveform. Its normalized
RMSE was 1.67710831e-05 and maximum absolute error was 8.2552433e-06, within the fixed limits of
1e-3 and 2e-3.

The native file uses the C-generated seeded excitation, native source STFT, and the complete C
pipeline. Its checks establish exact duration, expected shape, finite samples, and bitwise repeat
stability for the same seed. This does not claim free-running sample parity with the reference.
Ticket 14 owns human listening acceptance.

Model and reference provenance:

- Kokoro-82M v1.0 model revision: `f3ff3571791e39611d31c381e3a41a3af07b4987`.
- Official inference source revision: `dfb907a02bba8152ca444717ca5d78747ccb4bec`.
- Checkpoint SHA-256: `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`.
- `af_heart` voice SHA-256: `0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff`.
- Converted C weight pack SHA-256: `05f6359815680a2635b3ca4479e3d775735fd5dac45cd7bdd23f76b04b9aac2e`.
- Fixture provenance is recorded in `reference/fixtures/manifest.json`.

WAV hashes:

- `09-af-heart-hello-controlled.wav`: `4c11afcd40d4021c71186a55b3eb64995132754e911d8e5d34dd05a51ccadbe5`.
- `09-af-heart-hello-native.wav`: `15aa8d689ae13abbc65e7ddac3ed5281bc07b6dffa3ec3ef7b31a78742c38a2d`.
