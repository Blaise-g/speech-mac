# Native and CoreML Kokoro comparison

Captured: 2026-10-12T11:39:44+00:00
Host: Apple M5, 16 GiB, macOS 27.0.1 (26A434).
Installed CoreML CLI `0.11.2` at `/opt/homebrew/bin/kokoro`; executable SHA-256 `0cb1c8dbc0b77da34c37305b7a825a6df1c9203e51149d9678243f44aac68225`; Homebrew formula SHA-256 `233f336b9e46b38f07ecb4899a4404dec2281184dd4fb67317fc873becb76fa4`. Source commit `6bfc02afb237a311caee927482e58c028edb6acb`. Installed executable reports 0.11.2; v0.11.2 source still declares 0.10.0 in KokoroSay.swift.
Native commit `17516f71d376c037c2fcb105a6fabc44ac374a83`; CLI SHA-256 `b2392406e9c74b3594df72fc0b7320d35ccc61087ef0ac7864aeb287837e27c5`; driver SHA-256 `98f2392e6f7c6790f1660b1ca304d4027087551f7c46033071c4ec88ef1edc17`; build inputs SHA-256 `75ae266369ab960703d7fc56631214de9577f57b50714c1ca770a00dd22a680b`; model pack SHA-256 `05f6359815680a2635b3ca4479e3d775735fd5dac45cd7bdd23f76b04b9aac2e`. Native pack is 327,624,400 bytes (327,575,880 float32 weight bytes) and includes the tested `af_heart` style. Build flags/compiler: `{'build_directory': '/Volumes/Code/Personal/kokoro-mac/.worktrees/kokoro-native-planning/.build', 'compiler_command': 'xcrun clang', 'compiler': 'Apple clang version 21.0.0 (clang-2100.3.34.2)', 'base_cflags': '-std=c17 -Wall -Wextra -Wpedantic -Werror', 'opt_cflags': '-O2', 'cli_preprocessor_flags': '-Isrc', 'native_driver_preprocessor_flags': '-Isrc -DKOKORO_BENCHMARK_TIMING', 'link_flags': '-framework Accelerate', 'benchmark_define': 'KOKORO_BENCHMARK_TIMING'}`.
CoreML model tag `models-2026-03-23` contains 65 files (110,755,755 bytes), with 54 separate voice assets; `af_heart` voice asset is 524,290 bytes.

Both tools use `af_heart`, speed `1.0`, explicit output WAV paths, and no playback. English text uses native eSpeak NG text-to-speech: 1.52.0  Data at: /opt/homebrew/Cellar/espeak-ng/1.52.0/share/espeak-ng-data versus CoreML's bundled BART G2P. Both outputs are mono PCM16 at 24 kHz; native writes WAV through its C writer and CoreML through AVAudioFile. The same prepared IPA characters are sent to both direct CLI paths, but each uses a separate tokenizer/model pipeline and does not guarantee identical model token IDs.
For the shared IPA string `həlˈoʊ fɹʌmðə tˈɜːmɪnəl`, the measured audio duration differed (see table). CoreML source applies adaptive BOS/EOS trims and output processing after tokenization; native has its own chunk/tokenizer path without that same trim step. These differences can affect duration, but this run did not isolate the cause or quantify each contribution. English and IPA results are implementation comparisons, not equal-duration or audio-quality claims.
Fresh process means a new CLI process; it does not mean cold model or hardware caches. No caches were purged.

## Latency outcomes

A faster outcome is reported only when observed ranges are disjoint; each side shows its own sample count.

| Workload | Path | Mode | Native median [min, max], n | CoreML median [min, max], n | Outcome |
| --- | --- | --- | --- | --- | --- |
| hello_from_terminal | Fresh CLI completion | english | 0.492 [0.489, 0.529] s, n=5 | 4.229 [4.198, 4.479] s, n=5 | native_faster |
| hello_from_terminal | Fresh CLI completion | ipa | 0.480 [0.470, 0.491] s, n=5 | 4.407 [4.310, 4.665] s, n=5 | native_faster |
| hello_from_terminal | Resident waveform | english | 0.263 [0.258, 0.277] s, n=25 | 0.296 [0.290, 0.296] s, n=5 | native_faster |
| representative_paragraph | Fresh CLI completion | english | 1.729 [1.680, 1.824] s, n=5 | 5.419 [5.356, 5.496] s, n=5 | native_faster |
| representative_paragraph | Resident waveform | english | 1.500 [1.450, 1.698] s, n=25 | 1.406 [1.369, 1.431] s, n=5 | coreml_faster |

Resident latency compares native driver's complete `request_to_waveform_seconds` with CoreML's daemon CLI-reported synthesis time rounded to milliseconds; both exclude WAV writing. Native has 25 measured warm requests per workload (5 sessions × 5); CoreML has 5 resident requests after one unmeasured setup request per workload.

## Audio output and process resources

Throughput is generated audio seconds per wall second for Fresh CLI and per request-to-waveform second for Resident. Native resident time includes its frontend and inference; CoreML's pinned v0.11.2 `synthesisTime` starts before text chunk preparation and ends after waveform post-processing, so it includes its text frontend and inference. Both exclude WAV writing. CPU is user plus system time. `/usr/bin/time -l` prints CPU to 0.01 s; a displayed 0.00 is recorded as below resolution, not as zero. CoreML resident client and service measurements are reported separately.

| Workload | Path | Mode | Tool | Audio seconds | Audio throughput | CPU seconds and scope | RSS and scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hello_from_terminal | Fresh CLI | english | native | 1.775 [1.775, 1.775] s, n=5 | 3.607 [3.356, 3.630] ×, n=5 | 0.470 [0.450, 0.490] s, n=5; direct process | 0.569 [0.568, 0.575] GiB, n=5; OS peak RSS |
| hello_from_terminal | Fresh CLI | english | coreml | 1.485 [1.485, 1.485] s, n=5 | 0.351 [0.332, 0.354] ×, n=5 | 4.010 [3.940, 4.130] s, n=5; direct process | 1.569 [1.569, 1.571] GiB, n=5; OS peak RSS |
| hello_from_terminal | Fresh CLI | ipa | native | 1.775 [1.775, 1.775] s, n=5 | 3.700 [3.613, 3.775] ×, n=5 | 0.460 [0.450, 0.470] s, n=5; direct process | 0.569 [0.567, 0.569] GiB, n=5; OS peak RSS |
| hello_from_terminal | Fresh CLI | ipa | coreml | 1.400 [1.400, 1.400] s, n=5 | 0.318 [0.300, 0.325] ×, n=5 | 4.170 [4.080, 4.300] s, n=5; direct process | 1.564 [1.531, 1.567] GiB, n=5; OS peak RSS |
| hello_from_terminal | Resident | english | native | 1.775 [1.775, 1.775] s, n=25 | 6.760 [6.416, 6.884] ×, n=25 | 2.050 [2.050, 2.080] s, n=5; driver lifetime incl load and warm-up | 0.570 [0.570, 0.573] GiB, n=5; driver OS peak RSS |
| hello_from_terminal | Resident | english | coreml | 1.485 [1.485, 1.485] s, n=5 | 5.017 [5.017, 5.121] ×, n=5 | client <0.010 s (below time -l resolution), n=5; daemon 0.036 [0.031, 0.041] s, n=5 | client peak 0.013 [0.013, 0.013] GiB, n=5; daemon sampled/request 1.120 [1.120, 1.121] GiB, n=5 |
| representative_paragraph | Fresh CLI | english | native | 9.825 [9.825, 9.825] s, n=5 | 5.681 [5.387, 5.848] ×, n=5 | 1.700 [1.650, 1.720] s, n=5; direct process | 0.667 [0.665, 0.668] GiB, n=5; OS peak RSS |
| representative_paragraph | Fresh CLI | english | coreml | 8.515 [8.515, 8.515] s, n=5 | 1.571 [1.549, 1.590] ×, n=5 | 4.220 [4.210, 4.300] s, n=5; direct process | 1.602 [1.566, 1.604] GiB, n=5; OS peak RSS |
| representative_paragraph | Resident | english | native | 9.825 [9.825, 9.825] s, n=25 | 6.551 [5.786, 6.775] ×, n=25 | 10.630 [10.420, 11.340] s, n=5; driver lifetime incl load and warm-up | 0.667 [0.664, 0.672] GiB, n=5; driver OS peak RSS |
| representative_paragraph | Resident | english | coreml | 8.515 [8.515, 8.515] s, n=5 | 6.056 [5.950, 6.220] ×, n=5 | client 0.010 [0.010, 0.010] s, n=1; 4 below resolution; daemon 0.105 [0.094, 0.121] s, n=5 | client peak 0.015 [0.015, 0.015] GiB, n=5; daemon sampled/request 1.160 [1.158, 1.160] GiB, n=5 |

### Audio-throughput outcomes

Throughput uses output duration as the numerator; higher is better. This is separate from the latency table.

| Workload | Path | Mode | Native median [min, max], n | CoreML median [min, max], n | Outcome |
| --- | --- | --- | --- | --- | --- |
| hello_from_terminal | Fresh CLI | english | 3.607 [3.356, 3.630] ×, n=5 | 0.351 [0.332, 0.354] ×, n=5 | native_higher_audio_throughput |
| hello_from_terminal | Fresh CLI | ipa | 3.700 [3.613, 3.775] ×, n=5 | 0.318 [0.300, 0.325] ×, n=5 | native_higher_audio_throughput |
| hello_from_terminal | Resident | english | 6.760 [6.416, 6.884] ×, n=25 | 5.017 [5.017, 5.121] ×, n=5 | native_higher_audio_throughput |
| representative_paragraph | Fresh CLI | english | 5.681 [5.387, 5.848] ×, n=5 | 1.571 [1.549, 1.590] ×, n=5 | native_higher_audio_throughput |
| representative_paragraph | Resident | english | 6.551 [5.786, 6.775] ×, n=25 | 6.056 [5.950, 6.220] ×, n=5 | inconclusive_overlapping_observed_ranges |

CoreML daemon lifetime CPU from ready through the last measured request was 1.587 s (includes two setup requests and excludes CLI clients). Its interval-sampled RSS peak was 1.498 GiB from 246 readings at 0.05 s during setup/request windows; gaps were unsampled. OS lifetime peak physical footprint was 1.018 GiB; it is not RSS.

## Measurement limits

`/usr/bin/time -l` CPU and peak RSS apply to the directly timed process. Native resident process totals include model load, first request, warm-up, repeated requests, and WAV writing. CoreML resident client resources are distinct from daemon per-request CPU deltas and interval-sampled RSS. OS-reported peak RSS for the daemon is unavailable; exact GPU/ANE attribution and memory accounting are also unavailable. CoreML backend requests `.all` compute units and its compiled metadata reports mixed precision/storage; native uses float32 weights through Accelerate. No listening judgment or audio-quality winner is claimed.
CoreML daemon CPU is the daemon PID's user+system process CPU, excluding the CLI client. `proc_pid_rusage`'s rusage v4 CPU counters are Mach time units; they are converted with the recorded host `mach_timebase_info` numerator/denominator. This process CPU does not attribute GPU/ANE execution.

CoreML direct IPA is supported, but resident IPA is unsupported because the daemon request contains text, voice, and speed; `--ipa` bypasses the daemon.

Global winner: not reported; latency results differ by workload/path and throughput is a separate measure.

## Rerun

```sh
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --offline --no-sync --locked python benchmarks/compare.py --fresh-runs 5 --resident-runs 5 --warm-repetitions 5
```

Commands, full stdout/stderr, per-sample provenance, WAV hashes/formats, and raw timings are in `raw.json`. Smoke/setup notes and the exact actual benchmark command are also retained there.

Presentation regenerated from the saved raw samples after capture; no benchmark was rerun for this report update.

Post-capture resource correction: the original raw capture is preserved as `raw-captured-before-cpu-unit-correction.json` (SHA-256 `7a4a075dd8618c7110ee8dcb1e977a27ce67ff0f7b3cce186da199bbdab85a74`). The corrected `raw.json` multiplies the two CoreML service CPU metrics by 125/3 from `mach_timebase_info`, reparses `/usr/bin/time -l` CPU fields from saved stderr so displayed 0.00 values are below resolution, and renames sampled `cpu_user_ns`/`cpu_system_ns` fields to `cpu_user_mach_ticks`/`cpu_system_mach_ticks` without changing their tick values. Capture runner SHA-256: `d8a65318e8cecb2af795a6f576117c5e9bea44171eb425006bec670232d285bf`; report-generation runner SHA-256: `fabb647c7c2508295c68cf97d2093f23c6cc123985783c932136444d99c4bef0`. Timing samples were unchanged and no inference was rerun.
