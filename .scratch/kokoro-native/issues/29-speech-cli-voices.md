# 29: Ship speech-mac tasks and compatible voices

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

CLI/main, WAV/playback adapters, voice catalog/schema/compatibility resolution, engine consumers, speech-mac build product, CLI tests/docs.

## What to build

speech-mac say positional/file/stdin, valid mono24k PCM16 WAV, completed-WAV default playback, help/version, installed named/default and explicit compatible local voice artifacts. Validate en/seed/--duration-delta with native meaning and only tested supported controls. Actionable failures for old voice names, --ipa and --speed/-s; no silent reinterpretation. No cloud voice creation, recording encoder or runtime network requirement. Preserve temp permissions/cleanup; tests stub playback or write audition WAVs, no unsolicited speech. Legacy remains separately checkable until31, but no final compatibility alias.

## Blocked by

28-resident-engine

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] speech-mac say positional/file/stdin, valid mono24k PCM16 WAV, completed-WAV default playback, help/version, installed named/default and explicit compatible local voice artifacts
- [x] Validate en/seed/--duration-delta with native meaning and only tested supported controls
- [x] Actionable failures for old voice names, --ipa and --speed/-s
- [x] no silent reinterpretation
- [x] No cloud voice creation, recording encoder or runtime network requirement
- [x] Preserve temp permissions/cleanup
- [x] tests stub playback or write audition WAVs, no unsolicited speech
- [x] Legacy remains separately checkable until31, but no final compatibility alias
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution and verification — 2026-10-25

- Wave base: `08a2f8bc6d7912e0df00b695e216abb32f0c006c`; branch `feat/phonon-migration`, existing run worktree.
- Worker `/root/29_speech_cli_voices`, attempt 1, implement-lite / GPT-6.1 Sol / low. No formal repair follow-up was needed.
- Result and verified source SHA: `2f994e469a86e64f78f988850742ea76faa6253e`, with `Ticket: 29-speech-cli-voices`. The Coordinator confirmed the trailer, clean tree and all 13 paths within the original or approved test-tool Footprint. No numerical kernels, dependencies, shared source assets or tracker files changed in the source commit.
- Atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/29-speech-cli-voices.report.md`.

`make build` now produces separate `speech-mac` and legacy `kokoro` executables with correct main filtering. The new CLI accepts one positional argument, UTF-8 file or stdin, writes mono 24 kHz PCM16 WAV, and defaults to playing a completed private temporary WAV through macOS afplay. Help/version, local compatible voice listing and named/default/explicit selection work. Controls are English, decimal uint64 seed and finite native duration_delta. Removed IPA/speed/-s/old-voice requests fail with guidance; no speed conversion occurs. `--` preserves literal option-shaped text. Output and cleanup failures propagate. The old executable remains separately checkable until ticket 31; no final compatibility alias was added. [README](../../../README.md) documents the maintained syntax and interim legacy boundary.

Voice kind is explicit. Raw F32 speaker_wavs `[512,T]` or `[1,512,T]` retains signature/epoch validation and one projection. Projected emb/audio_prompt `[T,512]` or `[1,T,512]` requires matching model_ext and uses no projection; no raw sig/epoch is required for that form, and no unbound projected fallback is accepted. T is 1..4096; data must be finite. The local catalog uses safe filenames and validated artifact metadata. Voice compatibility `ba7a5a45@300` is the target config's embeddings_ext, distinct from full model identity `8aa279a6@100`. Explicit conflicts and ambiguous/multiple tensor forms fail.

### Worker Question and derived test-tool Footprint

Q29-1: the initial independent projected voice existed only under `.build`. The Coordinator approved `reference/phonon/export_projected_voice.py`, its two focused tests and reference README commands so another checkout can reproduce the model gate. Source: the approved compatible-voice/schema/integration-test requirements; no new runtime feature or acceptance change.

The exporter verifies pinned assets, reads the published GGUF BF16 speaker projection and pinned raw default voice, applies independent Torch F.linear, and saves an emb artifact with model_ext-only metadata plus provenance. It reads no native output and changes no frozen fixture or budget. The derived artifact SHA-256 is `0698fc2ca432d4f490559d1734c93f35f4e27045f594b300c0dcaaaed19d770e`; provenance records checkpoint/config/raw-voice hashes, model_id, embeddings_ext, weight dtype and output shape. The documented locked-uv command was executed; independent weight-orientation mutation fails its arithmetic/layout test.

### Independent Coordinator verification

Passed `/private/tmp/speech-mac-29-coordinator-verify.sh`, exit 0, against the committed source:

- Regenerated the independent projected artifact and confirmed the SHA above; passed its two locked-uv tests.
- Passed actual CLI integration under ASAN/UBSAN, including positional/file/stdin/default/named/local/projected selection, native controls, malformed UTF-8, completed-playback stub, private temporary permissions and cleanup/error paths. Voice parser ASAN/UBSAN checks passed separately; no sanitizer diagnostics occurred.
- Passed all five preserved actual engine cases at O3 with exact samples/chunks and batch/callback equality: 30720/1, 49920/1, 72960/1, 503040/2, 1242240/2. Worst engine PCM NRMSE remained 2.74799634e-5, max absolute error 8.28951597e-5, within unchanged 0.001/0.002 gates.
- Passed complete `make phonon-native-check` with actual Flow/condition fixtures selected: existing prefix/39-step consumers, sampler and 209-tensor/36-Q8 projection checks, plus the frozen autonomous generation/PCM verifier. Original 32-frame head/conditions/EOS/latents/PCM remain exact zero with 61440 raw/final samples.
- Passed `make format` then canonical `make check` with the real CLI assets selected. Older model checks skipped there are separately covered by the focused actual engine/native gates above or explicitly labelled as unrelated legacy checks.
- Invoked built `speech-mac --help`, `--version`, and actual `say 'Hello.' --seed 123 --output`. The WAV is byte-identical to the public CLI integration output; Python's independent wave reader confirmed mono PCM16/24 kHz and 30720 samples. No playback was invoked.
- Confirmed clean source tree and `git diff --check`. Logs: `/private/tmp/speech-mac-29-coordinator-{export,reference,cli-sanitizer,voice-sanitizer,engine-optimized,preserved,format,check,help,version}.log`. The owned audition output is ignored `.build/speech-cli-coordinator.wav`.

The worker additionally passed actual CLI normal/O2/ASAN/O3, the regenerated-artifact gate, all 16 locked Python reference tests (including frozen default/Q8/Mimi replays), the full suite and staged-snapshot hook. Temporary-copy mutations prove overflow, end-options, output errors, input bounds, WAV rate, projected kind, double projection and ignored unlink failures are detected. A fake compiler proved the old build recipe hid the first failure (exit 0), while the corrected recipe propagates it (exit 2). Every mutation was reversed and source hashes retained.

### Review disposition

One GPT-6.1 Sol / low Single Reviewer examined all 13 changed paths with Correctness, Standards, Spec, Reuse, Quality and Efficiency lenses; final refresh found no remaining findings or omissions.

| Id | Finding | Disposition and evidence |
| --- | --- | --- |
| C1 | Temporary unlink failures were ignored | Applied: report failed path/error and fail; actual cleanup regression and mutation |
| S1 | Weightless catalog/input/WAV coverage was missing | Applied: compatible synthetic catalog, input/error boundaries and independent WAV contracts run without model assets |
| R1 | Exporter duplicated the whole-file hash helper | Applied: reuse streaming quantization.sha256; exporter/unit/actual regeneration checks |
| K1 | Voice header still claimed projected forms were unsupported | Applied: accurate representation/layout/ownership contract |
| K2 | First build failure could be hidden by second compiler success | Applied: set -e; fake-compiler red/green and full build |
| K3 | Missing end-of-options support lost literal text behavior | Applied: parser/docs plus semantic regression/mutation |
| K4 | Help/version/voice listing could hide output failures | Applied: flush/error propagation; reachable read-only FILE tests and mutation |

The seven existing analyzer warnings remain visible (three normalizer, four unused unmodified SLEEF functions), with no new warning or suppression. Physical afplay/device playback and listening were not performed; tests use playback stubs or output files and play no sound. Streaming/device functional work belongs to ticket 30, and listening remains explicitly separate. No required ticket 29 criterion remains open. Final whole-branch GPT-6.1 Sol / high review still follows ticket 35.
