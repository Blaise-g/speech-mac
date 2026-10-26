"""Bound slash-word scanning by deterministic bytes examined, not elapsed time."""
import ctypes as C
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class SlashWordWork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        wrapper = root / "normalizer.c"
        wrapper.write_text('''#include <string.h>
#include <stdlib.h>
static size_t scanned;
static size_t measured_strlen(const char *s) {
    size_t n = strlen(s);
    scanned += n + 1;
    return n;
}
#define strlen measured_strlen
#include "''' + str(ROOT / 'src/english_normalize.c') + '''"
#undef strlen
char *slash(const char *s) {
    bool failed = false;
    scanned = 0;
    return rewrite_slash_word(s, &failed);
}
size_t work(void) { return scanned; }
void release(char *s) { free(s); }
''')
        library = root / "normalizer.dylib"
        subprocess.run(["xcrun", "clang", "-std=c17", "-O2", "-dynamiclib", "-I" + str(ROOT / "src"),
                        str(wrapper), str(ROOT / "src/utf8.c"), str(ROOT / "src/speech_error.c"),
                        "-o", str(library)], check=True)
        cls.lib = C.CDLL(str(library))
        cls.lib.slash.argtypes = [C.c_char_p]
        cls.lib.slash.restype = C.c_void_p
        cls.lib.work.restype = C.c_size_t
        cls.lib.release.argtypes = [C.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_linear_ascii_and_multibyte_slash_words(self):
        for unit in ("ab", "éß"):
            for count in (8192, (1048576 - 2) // (2 * len(unit.encode()))):
                word = unit * count
                source = (word + "/" + word + ".").encode()
                result = self.lib.slash(source)
                try:
                    self.assertTrue(result)
                    self.assertEqual(C.string_at(result), (word + " " + word + ".").encode())
                    # Constant work from small rule tables plus linear scans.
                    self.assertLess(self.lib.work(), 128 * len(source) + 100000)
                finally:
                    self.lib.release(result)

    def test_slash_boundaries(self):
        for source, expected in (("ab/cd", "ab cd"), ("éé/ßß!", "éé ßß!"),
                                 ("a/cd", None), ("ab/", None), ("/cd", None),
                                 ("km/hr", None), ("ab/12", None)):
            result = self.lib.slash(source.encode())
            try:
                self.assertEqual(C.string_at(result).decode() if result else None, expected)
            finally:
                self.lib.release(result)
