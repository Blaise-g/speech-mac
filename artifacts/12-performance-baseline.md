# Native Kokoro performance baseline

Captured: 2026-10-12T09:51:12+00:00

Hardware: Apple M5, 16 GiB, macOS 27.0.1 (26A434).
Source commit: 832032b97fb514848064b3498e9bba0b041ec1f6; source SHA-256: da0d0eb8e688746bf2878317d728fdfe21a7873b6c6d95a9195676f3c0cf6614.
Build: xcrun clang (Apple clang version 21.0.0 (clang-2100.3.34.2)); -std=c17 -Wall -Wextra -Wpedantic -Werror -O2; CLI flags -Isrc; native driver flags -Isrc -DKOKORO_BENCHMARK_TIMING; link flags -framework Accelerate.
Build output directory: /Volumes/Code/Personal/kokoro-mac/.worktrees/kokoro-native-planning/.build.
Model revision: f3ff3571791e39611d31c381e3a41a3af07b4987; pack 327,624,400 bytes; 327,575,880 float32 weight bytes; pack SHA-256 05f6359815680a2635b3ca4479e3d775735fd5dac45cd7bdd23f76b04b9aac2e.
Voice: af_heart style embedded in the model pack; official voice SHA-256 0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff; speed 1.0; seed 104729.

Each entry gives median, [min, max], and (max - min) / median variation. No performance target is assumed.

| Workload | Cold CLI end-to-end | Cold process to full waveform | Engine create | Frontend | Warm request to full waveform | Native synthesis | WAV close | Audio duration | Audio throughput | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_assistant_reply (english) | 421.39 ms [417.64, 713.05] (70.1%, n=5) | 416.37 ms [411.73, 596.82] (44.5%, n=5) | 183.58 ms [183.10, 193.23] (5.5%, n=5) | 8.95 ms [8.78, 9.63] (9.5%, n=25) | 222.58 ms [219.50, 230.42] (4.9%, n=25) | 213.32 ms [210.69, 221.10] (4.9%, n=25) | 0.78 ms [0.71, 4.62] (502.1%, n=25) | 1.70 s [1.70, 1.70] (0.0%, n=25) | 7.97 × [7.69, 8.07] (4.8%, n=25) | 0.13 × [0.12, 0.13] (4.9%, n=25) |
| paragraph (english) | 1466.85 ms [1448.44, 1592.77] (9.8%, n=5) | 1459.59 ms [1445.91, 1505.90] (4.1%, n=5) | 186.49 ms [185.54, 188.07] (1.4%, n=5) | 11.84 ms [11.55, 13.45] (16.0%, n=25) | 1261.88 ms [1239.93, 1363.98] (9.8%, n=25) | 1250.32 ms [1228.09, 1351.58] (9.9%, n=25) | 3.85 ms [3.60, 6.10] (64.9%, n=25) | 9.82 s [9.82, 9.82] (0.0%, n=25) | 7.86 × [7.27, 8.00] (9.3%, n=25) | 0.13 × [0.12, 0.14] (9.9%, n=25) |
| short_assistant_reply_prepared_phonemes (ipa) | 410.91 ms [410.26, 412.41] (0.5%, n=5) | 406.66 ms [406.03, 411.68] (1.4%, n=5) | 186.93 ms [186.81, 188.02] (0.6%, n=5) | 0.00 ms [0.00, 0.01] (233.3%, n=25) | 212.71 ms [211.51, 214.39] (1.4%, n=25) | 212.71 ms [211.51, 214.39] (1.4%, n=25) | 0.78 ms [0.73, 0.84] (14.1%, n=25) | 1.70 s [1.70, 1.70] (0.0%, n=25) | 7.99 × [7.93, 8.04] (1.4%, n=25) | 0.13 × [0.12, 0.13] (1.4%, n=25) |

Warm native synthesis stage medians:

| Workload | Largest measured stage | Share of median native synthesis | Stage medians |
| --- | --- | ---: | --- |
| short_assistant_reply (english) | neural_generator | 87.8% | context_encoder: 10.10 ms [9.84, 10.86] (10.1%, n=25), duration_prediction: 2.28 ms [2.24, 2.54] (13.1%, n=25), pitch_noise_prediction: 3.57 ms [3.51, 3.77] (7.2%, n=25), text_encoder: 1.01 ms [0.99, 1.22] (23.1%, n=25), decoder_conditioning: 7.55 ms [7.44, 8.02] (7.6%, n=25), neural_generator: 187.30 ms [185.21, 195.31] (5.4%, n=25), istft: 1.12 ms [1.09, 1.25] (14.3%, n=25) |
| paragraph (english) | neural_generator | 88.9% | context_encoder: 55.45 ms [54.81, 59.04] (7.6%, n=25), duration_prediction: 13.73 ms [13.41, 14.12] (5.2%, n=25), pitch_noise_prediction: 18.88 ms [18.71, 19.33] (3.3%, n=25), text_encoder: 5.67 ms [5.59, 7.43] (32.4%, n=25), decoder_conditioning: 36.37 ms [35.69, 38.57] (7.9%, n=25), neural_generator: 1111.58 ms [1091.74, 1209.54] (10.6%, n=25), istft: 6.91 ms [6.47, 7.40] (13.5%, n=25) |
| short_assistant_reply_prepared_phonemes (ipa) | neural_generator | 87.8% | context_encoder: 10.10 ms [9.94, 10.72] (7.7%, n=25), duration_prediction: 2.27 ms [2.24, 2.44] (9.0%, n=25), pitch_noise_prediction: 3.57 ms [3.53, 3.78] (7.0%, n=25), text_encoder: 1.01 ms [1.00, 1.06] (6.0%, n=25), decoder_conditioning: 7.57 ms [7.42, 8.36] (12.3%, n=25), neural_generator: 186.77 ms [185.27, 188.84] (1.9%, n=25), istft: 1.22 ms [1.10, 1.31] (17.7%, n=25) |

Process and model-load probes:

- Fresh CLI process (--version): 4.34 ms [3.26, 166.89] (3774.2%, n=5)
- Model pack open plus full payload SHA-256 validation: 107.66 ms [107.29, 107.98] (0.6%, n=5)

Engine creation includes pack open and validation, voice validation, and stage setup. Each first-request sample starts a new native driver process; cold refers to process and engine state, while OS file-page-cache state is uncontrolled. Warm repetitions reuse that session's resident engine. The separate pack-open probe runs after synthesis sessions, is not subtracted from engine creation, and does not estimate cold storage I/O. No cache purge was used.

First audio means the complete in-memory mono waveform is available after all chunks have been synthesized and combined. The current engine does not stream. WAV time is the writer returning after file close; it does not measure fsync or durable-media flush.

Audio throughput is generated audio seconds divided by native synthesis seconds. RTF is native synthesis seconds divided by generated audio seconds. Warm request time includes the English frontend or IPA tokenizer, synthesis, and waveform combination; it excludes WAV writing.

Stage timings bracket the seven existing synthesis calls in a benchmark-only build. They add no runtime API or product-build instrumentation. Each stage adds two monotonic clock reads and one accumulator callback per chunk; that instrumentation is included in total synthesis time and the JSON other_synthesis_overhead residual, so stage shares are approximate.

Raw samples and complete provenance are in the adjacent JSON report.
