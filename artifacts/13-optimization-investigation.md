# Ticket 13 optimization investigation

Ticket 13 remains open. Profiling identified a candidate, but the sine substitutions tested either exceeded the fixed generator phase tolerance or had no demonstrated material speed benefit. No native code change or speedup claim is included.

The run used the clean starting source commit `e7c646536fc68911930d356d39630d618ca52e54` (native-tree SHA-256 `5d0ca38dbb1591da46330a09991f8fedbd3ed34b08742d42a7249e18f92da785`), Apple M5 with 16 GiB, macOS 27.0.1 (26A434), Apple clang 21.0.0, `-std=c17 -Wall -Wextra -Wpedantic -Werror -O2`, Accelerate, the ticket 12 model pack, voice `af_heart`, speed 1, and seed 104729. The runner used locked `uv` and 25 warm samples per workload. OS file-page-cache state was uncontrolled.

| Workload | Ticket 12 synthesis median and variation | Ticket 13 pre-change run median and range variation | Ticket 13 generator median |
| --- | --- | --- | --- |
| Short English reply | 213.32 ms [210.69, 221.10], 4.9% | 212.333 ms [207.561, 265.241], 27.2% | 184.646 ms [181.428, 238.475], 30.9% |
| English paragraph | 1250.32 ms [1228.09, 1351.58], 9.9% | 1272.535 ms [1253.431, 1403.623], 11.8% | 1128.815 ms [1113.815, 1246.184], 11.7% |

The ticket 12 stage timings put `neural_generator` at 87.8–88.9% of median synthesis. An 8-second `sample` of the task's own benchmark PID showed repeated `sinf` and `apply_adain` stacks under `run_resblock`; source STFT was not a hot stack, and its twiddle coefficients were already precomputed. This stack sample is qualitative, not a cycle profile. The raw sample report at `/private/tmp/13-generator-profile.txt` has SHA-256 `a1e231b98b0ff426340612f8aec28f3d194431d770577b57a38ad66c10a469ef`. The ticket 13 pre-change benchmark JSON at `/private/tmp/13-before.json` has SHA-256 `85bdade0380705b428e755039cfe56ca162fccaf17b325e4fddba1e41166c4e2`.

Replacing `sinf` with `vvsinf`, `__sincosf`, or batched double-precision `vvsin` missed the fixed spectral-phase budget at 1 of 72,611 values. The first miss was index 52,803; the allowed error was `1.87394089e-5`, while the tested differences ranged from `2.28872523e-5` to `3.43313441e-5`. A double-polynomial fast path through absolute argument `0.03` also failed. A cutoff of `0.02` passed the generator check; a cutoff of `0.01` passed the generator and controlled full-audio checks. The `0.03` cutoff failed, and no paired timing showed that the narrower paths improve throughput. They were discarded.

Adding `restrict` to the profiled AdaIN inputs and outputs preserved output behavior, but Clang's `-O2` loop-vectorization remarks showed the relevant AdaIN loops vectorized both with and without it. The change therefore had no code-generation basis and was discarded. No candidate received a paired optimized timing run because none was both a sound, measurable optimization and within the fixed phase budget.

The source tree is restored to the starting commit. The investigation found no safe, verified gain above ticket 12's 4.9% short-reply and 9.9% paragraph warm-synthesis variation. The ticket remains open; no correctness budget was changed.
