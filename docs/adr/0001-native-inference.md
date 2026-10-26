# Native C inference on Apple Silicon

Implement Kokoro-82M v1.0 inference directly in C for Apple Silicon macOS, beginning with Accelerate. This keeps the shipped inference path independent of Python and of an embedded model runtime while allowing reference and conversion tools in Python; consider Metal after profiling shows a reason to add it.


Historical decision, superseded for the model identity by [0002](0002-phonon-native-runtime.md). The native C and narrow platform-boundary rationale remains in force.
