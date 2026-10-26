# Phonon native runtime under speech-mac

Status: Accepted · 2026-10-26

## Context

The accepted runtime is Phonon 40M English with local text, voice, WAV and progressive playback workflows. Keeping a second model implementation would retain obsolete phoneme, style and speed contracts and their dependencies.

## Decision

Maintain speech-mac with one native C17 Phonon runtime on Apple Silicon macOS. Keep earlier model implementations and measurements in Git history. Keep reusable UTF-8, WAV, error and AudioQueue seams under speech names. Use Accelerate/NEON CPU operations; the runtime has no Metal path, plugin framework or embedded Python/Rust/Core ML runtime. Development references remain isolated in the locked `reference/phonon` project.

The public request API retains `SpeechEngine` and `SpeechStatus`. Shared helpers and playback use `SpeechIOStatus` and `SpeechError`, separating detailed I/O/parser failures from request outcomes. Compatible raw and projected voices retain distinct validation and projection semantics.

## Consequences

Unsupported options fail as invalid arguments before asset access. Incompatible or missing voices fail voice validation. Historical evidence retains its model identity, and the authoritative `.scratch/kokoro-native` tracker path remains. The local checkout is not moved. The GitHub repository is `Blaise-g/speech-mac`; its rename and publication records are in [remote setup](../remote-setup.md). A local consumer census cannot promise compatibility for undiscoverable private clients.
