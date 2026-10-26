"""Compiled consumer layout and supported archived-source boundary."""
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trace_native_operators as trace


class OperatorABI(unittest.TestCase):
    def test_current_and_compatible_archived_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A real committed archive, using the current compatible header ABI.
            archive = root / "archived.c"
            archive.write_bytes(subprocess.check_output(
                ["git", "show", "4b813bd8992000f9608b6bef90d2d18d67d36e11:src/phonon_transformer.c"], cwd=trace.ROOT))
            lib = trace.build_library(archive, root)
            lib.phonon_linear.argtypes = [C.POINTER(trace.TensorView), C.c_void_p,
                C.c_size_t, C.c_size_t, C.c_void_p, C.c_void_p]
            packed = (C.c_uint16 * 1)(0x4000)  # BF16 2.0
            cached = (C.c_float * 1)(3.0)
            x, out = (C.c_float * 1)(4.0), (C.c_float * 1)()
            view = trace.TensorView(2, 2, (C.c_uint64 * 8)(1, 1),
                C.cast(packed, C.c_void_p), 2, C.cast(cached, C.c_void_p))
            self.assertEqual(lib.phonon_linear(C.byref(view), x, 1, 1, None, out), 1)
            self.assertEqual(out[0], 12.0)  # Cached expansion must occupy the native field.
            attention = lib.phonon_prefill_attention_test
            attention.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]
            self.assertEqual(attention(None, None, None, 0, None), 0)

    def test_layout_drift_rejected_at_compile_before_tensor_calls(self):
        class OldView(C.Structure):
            _fields_ = trace.TensorView._fields_[:-1]
        class ReorderedView(C.Structure):
            _fields_ = [trace.TensorView._fields_[1], trace.TensorView._fields_[0],
                        *trace.TensorView._fields_[2:]]
        for view in (OldView, ReorderedView):
            with self.subTest(view=view), tempfile.TemporaryDirectory() as directory, patch.object(
                    trace, "TensorView", view):
                with self.assertRaises(subprocess.CalledProcessError):
                    trace.build_library(trace.ROOT / "src/phonon_transformer.c", Path(directory))

    def test_unsupported_private_seam_archive_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "old.c"
            archive.write_text("static int flow_attention(void) { return 0; }")
            with self.assertRaisesRegex(ValueError, "use the diagnostic from the archived source revision"):
                trace.build_library(archive, root)
