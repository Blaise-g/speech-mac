# Native C inference on Apple Silicon

Implement Kokoro-82M v1.0 inference directly in C for Apple Silicon macOS, beginning with Accelerate. This keeps the shipped inference path independent of Python and of an embedded model runtime while allowing reference and conversion tools in Python; consider Metal after profiling shows a reason to add it.
