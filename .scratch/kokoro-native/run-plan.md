# Kokoro native inference: approved Run Plan

Status: approved by the Coordinator on 2026-10-11.

## Run context

- Requirement Source: `.scratch/kokoro-native/spec.md` in `/Volumes/Code/Personal/kokoro-mac`.
- Run branch: `feat/kokoro-native`, based on setup commit `963ce48eb1da6f6b067588b067e6d325313bcf93`.
- Worktree: existing `/Volumes/Code/Personal/kokoro-mac/.worktrees/kokoro-native-planning`.
- Tracker: local files under `.scratch/kokoro-native/issues/`, with this approved plan and the Requirement Source committed on the run branch.
- Repository for every ticket: `/Volumes/Code/Personal/kokoro-mac`.
- Worker Profile for every implementation ticket: `implement-lite · xhigh · gpt-6-luna (override)`.
- Environment access: `none` for every ticket. No rollout ticket is included.
- Pre-approved actions: none. Push and pull request: no.
- Dependency graph: the ticket edges below express true prerequisites; ticket 05 only depends on 03, not 04. Execution schedule: all 14 tickets still run strictly one after another because the repository rule forbids parallel work within one repository, so ticket 05 will be scheduled after 04.
- Consumer census: none required. No ticket removes, renames, or deprecates an existing consumer-facing surface; there are no UI consumers.

The architecture boundaries below follow the official [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M), its [model config](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json), and the official [inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), [model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), and [ISTFTNet decoder](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py). The reference ticket pins immutable model, code, voice, and tool revisions before fixture generation. The model path is phoneme IDs plus a voice style through the contextual encoder, duration alignment, pitch/noise prediction, text encoder, and ISTFTNet to a 24 kHz waveform. The upstream project is used only for reference and conversion; shipped inference remains C.

## Ticket breakdown

All entries are in dependency order. Each ticket's context will cite `.scratch/kokoro-native/spec.md` as the product background and the pinned official model files as its technical source. Each executable slice adds defect-revealing checks at the highest stable C seam available, with oracle fixtures at the stage boundary. `make format`, then `make check`, and independent review remain required before each ticket commit.

### 01 — `01-reference-oracle-and-weight-pack`: Pin the reference and produce C-readable model data

- **Blocked by:** none.
- **What it delivers:** a repeatable way to create the one-voice v1.0 model artifact and authoritative prepared-phoneme reference outputs without placing Python in the shipped inference path.
- **Acceptance checks:** Pin immutable official model/config/source, one initial voice (`af_heart` proposed), and reference tool versions by revision/hash. Convert the checkpoint and voice style into a documented C-readable tensor pack. Generate compact fixtures for every planned module boundary and final waveform from fixed phoneme IDs. Record fixture provenance and shape/dtype metadata; keep model weights outside Git. Before native C work begins, establish per-stage and final-audio error metrics and budgets from reference repeatability plus a documented numerical-precision rationale; no native C output is used to choose those budgets.
- **Tests:** Regenerate reference outputs and compare to checked-in fixture data; validate tensor-pack names/shapes/metadata through the reference-side converter and reader; reject truncated, malformed, or incompatible pack inputs. Verify reference repeatability across runs. Record the budget rationale alongside fixture provenance. Keep C inference implementation out of this ticket so budgets are fixed first.
- **Ticket Key:** `01-reference-oracle-and-weight-pack`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 02 — `02-phoneme-context-encoder`: Produce Kokoro contextual phoneme features in C

- **Blocked by:** `01-reference-oracle-and-weight-pack`.
- **What it delivers:** prepared phoneme IDs and model weights produce contextual token features matching the official ALBERT/PL-BERT path, including its projection into Kokoro's hidden width.
- **Acceptance checks:** C loads the required weights from the pack, applies the v1.0 token/position embedding and attention stack, mask, and projection, and returns the expected feature shape for valid sequences. Overlong or invalid inputs fail clearly. No Python inference runtime is required.
- **Tests:** Compare stage outputs to the pinned fixture for short and boundary-bearing phoneme sequences using the pre-established budget; test mask/shape handling and invalid context length. A budget miss prompts investigation and a fix; any budget change needs independent numerical evidence and a written rationale, never merely observed native error.
- **Ticket Key:** `02-phoneme-context-encoder`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 03 — `03-duration-alignment`: Predict token durations and expand the alignment

- **Blocked by:** `02-phoneme-context-encoder`.
- **What it delivers:** contextual phoneme features and the predictor half of the voice style produce per-token durations and the frame-to-token alignment used by the decoder.
- **Acceptance checks:** Implement the v1.0 duration encoder, recurrent projection, speed scaling, rounding/clamping, and monotonic repeated-token alignment. Durations are positive; each output frame maps to exactly one input token; output length equals the sum of durations.
- **Tests:** Compare predictor features, durations, and alignment to oracle fixtures; exercise multiple token lengths and speed values; test alignment invariants independently of floating-point tolerance.
- **Ticket Key:** `03-duration-alignment`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 04 — `04-pitch-noise-prediction`: Predict aligned pitch and noise conditioning

- **Blocked by:** `03-duration-alignment`.
- **What it delivers:** duration-aligned prosody features and predictor style produce the F0 and noise conditioning sequences needed by ISTFTNet.
- **Acceptance checks:** Implement the v1.0 shared recurrent path, style-conditioned F0/noise blocks, and their projections. Returned sequences align in length with the duration-expanded frames and contain finite values.
- **Tests:** Compare F0/noise outputs at the module seam against fixtures using the pre-established budget; test sequence-length and finite-value invariants. A budget miss prompts investigation rather than automatic threshold widening.
- **Ticket Key:** `04-pitch-noise-prediction`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 05 — `05-phoneme-acoustic-encoder`: Encode phonemes and apply the predicted alignment

- **Blocked by:** `03-duration-alignment`.
- **What it delivers:** prepared phoneme IDs become text-encoder features expanded onto the predicted acoustic frame sequence.
- **Acceptance checks:** Implement the v1.0 embedding, convolutional/residual stack, recurrent layer, padding mask, and alignment multiplication. Output time dimension matches the duration expansion.
- **Tests:** Compare text-encoder and aligned acoustic features to oracle fixtures using the pre-established budget; test padding and alignment lengths. A budget miss prompts investigation rather than automatic threshold widening.
- **Ticket Key:** `05-phoneme-acoustic-encoder`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 06 — `06-decoder-conditioning`: Prepare style-conditioned decoder features

- **Blocked by:** `04-pitch-noise-prediction`, `05-phoneme-acoustic-encoder`.
- **What it delivers:** aligned text features plus F0/noise and decoder style become the conditioned feature sequence consumed by the waveform generator.
- **Acceptance checks:** Implement the pre-generator ISTFTNet conditioning and residual blocks with the official channel/time shapes and style split. Reject mismatched conditioning lengths instead of producing partial audio.
- **Tests:** Compare the pre-generator output to its oracle fixture using the pre-established budget; cover expected sequence lengths and mismatch errors. A budget miss prompts investigation rather than automatic threshold widening.
- **Ticket Key:** `06-decoder-conditioning`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 07 — `07-neural-waveform-generator`: Generate ISTFTNet spectral frames

- **Blocked by:** `06-decoder-conditioning`.
- **What it delivers:** decoder features, style, and pitch conditioning produce the learned upsampled spectral representation, including the periodic/noise excitation used by Kokoro v1.0.
- **Acceptance checks:** Implement the configured upsample stages, residual blocks, source excitation, and final spectral coefficients. Provide a deterministic test seam for the excitation/noise input so module comparisons are reproducible.
- **Tests:** Compare intermediate and final spectral coefficients against pinned fixtures using captured excitation inputs and the pre-established budget; verify the configured output frame count and finite values. A budget miss prompts investigation rather than automatic threshold widening.
- **Ticket Key:** `07-neural-waveform-generator`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 08 — `08-istft-waveform`: Reconstruct the 24 kHz waveform

- **Blocked by:** `07-neural-waveform-generator`.
- **What it delivers:** Kokoro spectral frames become a normalized waveform with the official inverse-STFT hop, window, and output sizing.
- **Acceptance checks:** Implement the v1.0 inverse-STFT reconstruction and return finite mono samples at 24 kHz with the length expected from the model output. Keep platform acceleration behind the narrow numeric operation seam where needed for reference checks.
- **Tests:** Compare waveform samples and output length to reference fixtures for fixed spectral frames using the pre-established budget; test edge/frame handling and non-finite rejection. A budget miss prompts investigation rather than automatic threshold widening.
- **Ticket Key:** `08-istft-waveform`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 09 — `09-phoneme-to-audio-correctness`: Verify complete direct C inference

- **Blocked by:** `08-istft-waveform`.
- **What it delivers:** prepared phonemes, one voice style, and speed synthesize complete Kokoro audio through the assembled native pipeline.
- **Acceptance checks:** The assembled inference path runs directly in C on Apple Silicon and matches the pinned official reference within the pre-established numerical budgets. Produce and retain a representative WAV audition artifact. This technical ticket does not wait for human listening; the final listening acceptance is tracked in ticket 14.
- **Tests:** Run full phoneme-to-waveform fixture comparisons with controlled excitation and pre-established budget; compare duration and waveform metadata; exercise the no-Python runtime path; record the audition artifact's location.
- **Ticket Key:** `09-phoneme-to-audio-correctness`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 10 — `10-resident-engine-phoneme-cli`: Reuse loaded weights and write phoneme-input WAVs

- **Blocked by:** `09-phoneme-to-audio-correctness`.
- **What it delivers:** a reusable C engine keeps model weights resident across requests, and the CLI can turn prepared phonemes into a valid WAV.
- **Acceptance checks:** Load model/voice once into an engine, synthesize repeated requests through that instance, and write mono 24 kHz WAV files with correct headers and sample counts. Document the model and voice artifact arguments.
- **Tests:** Call the engine repeatedly and compare outputs to the direct inference seam; exercise the CLI end to end with prepared phonemes; validate WAV fields and output length; check clear errors for missing or incompatible model artifacts.
- **Ticket Key:** `10-resident-engine-phoneme-cli`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 11 — `11-english-espeak-frontend`: Synthesize ordinary English through eSpeak NG

- **Blocked by:** `10-resident-engine-phoneme-cli`.
- **What it delivers:** ordinary English text is converted by the agreed eSpeak NG frontend and synthesized by the resident native engine through the CLI.
- **Acceptance checks:** Support the initial English path with documented eSpeak-to-Kokoro symbol mapping, while accepting prepared phonemes as the reference path. Reject empty/unmappable output clearly; document that pronunciation can differ from Misaki.
- **Tests:** Check representative English inputs against expected frontend tokens/IDs; run a text-to-WAV CLI check with eSpeak NG; verify output through the same WAV and audio checks. Keep eSpeak and Python out of C model inference itself.
- **Ticket Key:** `11-english-espeak-frontend`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 12 — `12-performance-baseline`: Measure cold start, warm synthesis, and first audio

- **Blocked by:** `11-english-espeak-frontend`.
- **What it delivers:** repeatable measurements show model-load cost, warm synthesis time, output duration, throughput, and latency on an identified Apple Silicon Mac.
- **Acceptance checks:** Report hardware/OS, build and model revisions, voice and input, cold start, warm request time, generated duration, and throughput. Because synthesis returns a complete waveform rather than streaming, define first-audio latency as time until that full in-memory waveform is available; report WAV write completion separately. Establish observed run-to-run variation before setting any optimization acceptance bound.
- **Tests:** Smoke-run the benchmark on phoneme and English input; verify timing fields and generated audio duration are populated and clearly labeled. No performance target is assumed in advance.
- **Ticket Key:** `12-performance-baseline`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 13 — `13-profile-guided-optimization`: Improve a measured bottleneck

- **Blocked by:** `12-performance-baseline`.
- **What it delivers:** one profile-backed change improves the measured native inference path while preserving reference correctness.
- **Acceptance checks:** Identify the dominant bottleneck from the baseline, make one targeted optimization, and report before/after measurements on the same Mac and workload. Accept a speed claim only when it exceeds the run-to-run variation observed in ticket 12. Re-run intermediate and final correctness checks. Do not add Metal preemptively; if profiling shows a Metal-suitable bottleneck, request a separate bounded ticket with evidence.
- **Tests:** Run full `make format` and `make check`, the affected stage fixtures, end-to-end waveform comparison, and paired benchmark runs using the same voice/input/device.
- **Ticket Key:** `13-profile-guided-optimization`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

### 14 — `14-listening-acceptance`: Record final human audio acceptance

- **Blocked by:** `13-profile-guided-optimization`.
- **What it delivers:** a human listening check on representative WAV output from the final optimized build confirms whether it is intelligible and free of obvious artifacts.
- **Acceptance checks:** Record which final-build WAV was auditioned and the listener's result. If no one can listen during the run, leave this final acceptance ticket explicitly blocked/pending; tickets 10–13 remain unblocked by that human check.
- **Tests:** No automated test can replace listening. Confirm the artifact was generated by the optimized build and record the manual result.
- **Ticket Key:** `14-listening-acceptance`.
- **Rollout:** no.
- **Worker Profile:** `implement-lite · xhigh · gpt-6-luna (override)`.

## Approved implementation decisions (appended to the Requirement Source)

- Implement the inference graph in readable C17, using Accelerate behind narrow numerical-operation seams when it serves the accepted Apple Silicon target. Do not embed a model runtime; Python remains development-only for reference, fixture, and weight conversion work.
- Use the official Kokoro-82M v1.0 config and checkpoint as the model contract. Pin immutable revisions and hashes in the reference/conversion work before producing fixtures. Start with one English voice (`af_heart` proposed) and prepared phonemes; keep eSpeak NG in the English frontend boundary.
- Treat the module boundaries in tickets 02–08 as the stable backend seams. Use a versioned C-readable weight pack produced from official weights, stored outside Git, and load it into a resident engine that accepts prepared phonemes, voice style, and speed and returns mono 24 kHz audio.
- Add no Metal path or abstraction in advance. A separate ticket can be proposed only if the measured profile demonstrates a suitable bottleneck and likely benefit.

## Approved testing decisions (appended to the Requirement Source)

- Use the official Python inference implementation as a development-time oracle, with deterministic prepared phoneme IDs, voice style, and captured excitation inputs. Keep compact checked-in fixtures for each module boundary and final audio; do not commit full checkpoint/voice weights.
- Before C inference implementation, set per-stage and final-audio numerical budgets from repeated-reference behavior and documented float-precision rationale. Test each C module at its stable input/output seam against oracle fixtures using those fixed budgets; a miss triggers diagnosis and a fix, with budget changes requiring independent evidence and written rationale. Test shape, masking, alignment, length, error, and finite-value invariants separately where exact numeric equality is not the right assertion.
- Test the assembled engine using the highest available seam: direct phoneme-to-waveform reference comparison, controlled excitation for reproducibility, repeat requests against one resident engine, and CLI WAV inspection. Ticket 09 records audition artifacts; ticket 14 holds final human listening acceptance so pending listening does not block technical follow-on tickets.
- Run `make format` followed by `make check` and an independent review before each commit. The English frontend gets token/ID tests and a CLI smoke test; benchmark and optimization tickets record the exact Mac/workload and use repeated paired measurements. Set any performance acceptance bound from observed baseline variation, never from an assumed number.

## Open items

- Coordinator approval of the granularity, issue edges, branch, and worktree is complete.
- The reference ticket will pin exact official source, model, voice, and tooling revisions, settle fixture format/size, and justify per-stage/final numerical budgets before C work. A later native mismatch must be investigated against those budgets rather than used to set them.
- No consumer census was triggered; there are no planned removals, renames, deprecations, or rendered UI changes.
- Metal is intentionally conditional and has no ticket in this base run. No remote push, pull request, rollout, or deployment is included.

## Run Plan table

| Order | Ticket Key | Blocked by | Repository | Branch ← base (evidence) | Worktree | Worker Profile | Environment access | Pre-approved actions | Push + PR | Rollout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 01-reference-oracle-and-weight-pack | None | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 02 | 02-phoneme-context-encoder | 01 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 03 | 03-duration-alignment | 02 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 04 | 04-pitch-noise-prediction | 03 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 05 | 05-phoneme-acoustic-encoder | 03 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 06 | 06-decoder-conditioning | 04, 05 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 07 | 07-neural-waveform-generator | 06 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 08 | 08-istft-waveform | 07 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 09 | 09-phoneme-to-audio-correctness | 08 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 10 | 10-resident-engine-phoneme-cli | 09 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 11 | 11-english-espeak-frontend | 10 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 12 | 12-performance-baseline | 11 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 13 | 13-profile-guided-optimization | 12 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |
| 14 | 14-listening-acceptance | 13 | kokoro-mac | feat/kokoro-native ← 963ce48 (verified setup HEAD) | existing: .worktrees/kokoro-native-planning (branch created from approved base) | implement-lite · xhigh · gpt-6-luna (override) | none; local work/read-only research | none | no | no |

Capacity: 4 concurrent threads total. Run one implementation worker plus its one reviewer with the coordinator (3 active threads). Same repository tickets execute sequentially; none run in parallel. Final listening requires a human result and stays open if unavailable.

Approval authorized publication of these tickets, appending implementation/testing decisions to the specification, creating the run branch at the verified setup commit in the existing planning worktree, committing the tracker, and subsequent local implementation/review commits. The run branch has been created. Remote publication is excluded.
