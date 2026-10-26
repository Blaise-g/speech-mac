#36: Public Mac setup and README

## Context

User2026-10-26: “how can i test it via command? also make the repo public and serf serveable on mac with a proper readme using $unslop”. Scope/Run Plan approved by Coordinator in active spec. Prior migration is merged in897b3da. Use implement-lite / GPT-6.1 Sol / low, final review Sol/high. Root is sole tracker/admin writer.

## Footprint

Makefile; scripts/ bootstrap helper(s); reference/phonon canonical setup/conversion/download helpers and tests only as needed; README.md, reference/phonon/README.md, CONTRIBUTING.md, docs/remote-setup.md; code LICENSE only after explicit user selection; setup-focused tests and CI wiring for those model-free tests. No inference C/math/fixtures/budgets/vendor/lock changes without a Worker Question. No historical artifact edits. Root-only GitHub visibility/description/readback and tracker.

## Acceptance criteria

- [x] `make bootstrap` prepares an Apple Silicon Mac checkout using installed Xcode CLI tools and uv: locked environment, verified pinned assets, canonical native pack, ignored default model/voice and release CLI. Existing `make setup` remains hook setup.
- [x] Prerequisites/unsupported platform/missing gated access fail clearly; HF credentials stay local; verified existing source assets can be reused offline; repeated setup is safe and idempotent; existing unrelated/custom files are preserved. No automatic audio/system/global environment changes.
- [x] Model-free tests expose setup failure/reuse/destination behavior; actual fresh-checkout setup and deterministic default-output WAV smoke pass using verified existing40M assets, without user speech playback.
- [x] README unslop explains prerequisites, fresh clone/auth/setup, explicit audible playback versus saved WAV, files/stdin/stream, voices/native controls, resource costs, troubleshooting and separate model/third-party/code licensing. Portable paths; no claim of unsupported languages/Intel/server/global install.
- [x] Independent changed-path review before Ticket36 commits; make format then make check; relevant locked setup tests; final Sol/high review and CI pass.
- [x] Root verifies public GitHub identity/history/settings/CI/anonymous read and updates stale repository description/current private-only documentation, after the redacted reachable-history credential audit.

Worker Profile: medium; implement-lite / gpt-6.1-sol / low.

Status: done

## Verified implementation and publication blocker, 2026-10-26

The source worker completed80545586a174a46efbd1e7266c475f58be90858f from clean286d9d9588d65ad409ae63cfe02bbbff69ce2e7c, Ticket36, with a clean tree. Exact nine-file footprint: Makefile, README, CONTRIBUTING, docs/cli-details.md, docs/remote-setup.md, reference/phonon/README.md and scripts/bootstrap.{sh,py}/tests/test_bootstrap.py. Implementation and one independent all-six-lens reviewer used GPT-6.1 Sol/low with implement-lite/write-tests/unslop; no findings. Original missing-target/helper red proof, nine model-free cases, format→check, canonical downloader test and staged hook passed. The initial unsynced reference-test invocation failed for absent dependencies; the correctly installed fresh environment passed, with both logs retained. No inference/math/fixture/budget/vendor/lock change.

The Coordinator independently verified the exact footprint/trailer/clean source, all nine bootstrap cases and canonical format→check. Exact committed source in a fresh spaced-path archive passed locked setup and an installed-environment offline repeat, preserving installed inode/mtime/bytes. An initial cold offline cache attempt lacked Torch; normal exact-lock dependency download resolved the setup environment, without a source change or global install. Default-path CLI output is mono PCM16/24k/76800 samples, byte-identical to the accepted short WAV (SHA256e1653a11cf7d39baf4a7b394ab727e995d802f9a74585bec50e1c28e0084faf2). The existing user checkout is bootstrapped and its default CLI/catalog work; original twelve assets and the models symlink/old files remain unchanged. No audio was played. Root report/data: /private/tmp/speech-mac-36-root-verify.log and speech-mac-36-root-result.json.

Independent final GPT-6.1 Sol/high SingleReviewer examined all eleven changed paths since merged897b3da, with all six lenses and unslop/factual retention checks. No source or overlap finding; the initially sequenced evidence gaps are T1(PR/main CI) and T2(final audit/public administration). The complete report is /private/tmp/speech-mac-36-high-review.report.md.

PR4 (https://github.com/Blaise-g/speech-mac/pull/4) has passed check at source80545586: https://github.com/Blaise-g/speech-mac/actions/runs/40341395505/job/121636744905. The final110-commit redacted Gitleaks scan found no credentials beyond21 individually reviewed tokenizer SHA256 false positives, excluded only by their exact historical fingerprints. All three existing PR heads are ancestors in the audited history; all PR text and eight retained CI logs were scanned with no credential. Private audit exports/reports remain outside Git.

Only the final public-administration criterion is blocked. Automatic approval review rejected both the read-only gh repo edit help invocation and the explicitly requested visibility/description command at process creation with: “approval required by policy, but AskForApproval is set to Never”. No mutation ran. The user's public authorization is already explicit; the environment cannot prompt for its required approval. A fresh readback still shows private=true and the old Kokoro description, with ADMIN permission. The user was given the exact combined gh repo edit visibility/description command to run in their own Terminal and asked to reply done. No alternate mutation path was attempted. The code/README/CI landing remains independent work to finish; live public/anonymous readbacks stay pending until the external setting changes.

The optional code-license choice remains unanswered, so no LICENSE or reuse grant was added. The README states that absence and separates model/upstream terms. Existing human listening/acoustic limits remain unchanged.

## Publication verified, 2026-10-26

The user reported completing the manual visibility/description command. Root confirmed `Blaise-g/speech-mac` is public with description “Native C17 Phonon 40M English text-to-speech for Apple Silicon macOS”, unchanged repository ID1415715156, default main, ADMIN permission, merge commits only and automatic merged-branch deletion. Main remains unprotected and no rulesets are configured; no protection change was requested or made. Anonymous API access confirms public identity, credential-helper-disabled Git access resolves HEAD/main to50baa092d6c301cd12eee6d29d8f1b6f32e4af06, and the anonymous raw README returns HTTP200.

PR4 is merged at50baa092d6c301cd12eee6d29d8f1b6f32e4af06. Its main CI run40346286812 completed successfully. The merged feature branch/worktree cleanup is complete; only the main worktree remains. T1 and T2 are resolved; the earlier automatic-review blocker above is historical. This completion record starts from50baa092d6c301cd12eee6d29d8f1b6f32e4af06 on docs/publication-verified and updates only tracker/publication documentation. The code-license choice remains optional and unanswered; listening and acoustic onset remain unverified.

For this documentation closeout, root ran make format then make check successfully; model-dependent checks without their selected fixtures remain explicit skips. The redacted reachable-history scan now covers111 commits and finds no leaks beyond the previously reviewed exact tokenizer-hash fingerprints. Logs: /private/tmp/speech-mac-36-publication-check.log and speech-mac-36-publication-secrets.log.
