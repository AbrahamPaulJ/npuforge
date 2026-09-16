"""Host allocator contract/stress tests; no models, Android build, or device needed."""
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(Path("/proc/self/maps").exists(), "requires Linux procfs")
class CompilerHeapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("gcc") or shutil.which("clang")
        cxx = shutil.which("g++") or shutil.which("clang++")
        if not cc or not cxx:
            raise unittest.SkipTest("C and C++ host compilers are required")
        cls.build = tempfile.TemporaryDirectory(prefix="compiler-heap-tests-")
        cls.addClassCleanup(cls.build.cleanup)
        directory = Path(cls.build.name)
        cls.library = directory / "libcompiler_heap.so"
        cls.probe = directory / "compiler_heap_probe"
        common = ["-O2", "-fno-builtin", "-Wall", "-Wextra", "-pthread"]
        # Android's LIBC symbol-version script would defeat glibc interposition.
        commands = [
            [cc, "-std=c11", *common, "-fPIC", "-c", str(ROOT / "native/compiler_heap.c"),
             "-o", str(directory / "heap.o")],
            [cxx, "-std=c++17", *common, "-fPIC", "-shared", str(directory / "heap.o"),
             str(ROOT / "native/compiler_new.cpp"), "-ldl", "-o", str(cls.library)],
            [cxx, "-std=c++17", *common, "-Werror", str(ROOT / "tests/compiler_heap_probe.cpp"),
             "-ldl", "-o", str(cls.probe)],
        ]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)

    def run_probe(self, scenario, *, crash=False):
        with tempfile.TemporaryDirectory(prefix="compiler-heap-backing-") as backing:
            environment = os.environ.copy()
            environment.update(LD_PRELOAD=str(self.library), QNN_COMPILER_HEAP_DIR=backing)
            result = subprocess.run([str(self.probe), scenario], env=environment,
                                    capture_output=True, text=True, timeout=25)
            self.assertEqual(result.returncode, -signal.SIGABRT if crash else 0,
                             result.stdout + result.stderr)
            if not crash:
                self.assertIn(f"{scenario} OK", result.stdout)
            self.assertEqual(list(Path(backing).iterdir()), [], "backing files must be unlinked")
            return result

    def test_tiny_objects_are_backed_and_pooled_with_random_reuse(self):
        self.run_probe("small")

    def test_sizes_calloc_and_realloc_across_old_cutoff(self):
        self.run_probe("boundaries")

    def test_alignment_zero_and_overflow_contracts(self):
        self.run_probe("alignment")

    def test_cpp_new_delete_variants(self):
        self.run_probe("cpp")

    def test_cross_thread_allocation_reallocation_and_free(self):
        self.run_probe("threads")

    def test_foreign_libc_allocations_keep_their_contents(self):
        self.run_probe("foreign")

    def test_crash_report_captures_mappings_and_preserves_signal(self):
        result = self.run_probe("crash", crash=True)
        counts = {}
        for name in ("memory_mappings", "scudo_secondary", "storage_backed"):
            match = re.search(rf"^{name}=0x([0-9a-f]+)$", result.stderr, re.MULTILINE)
            self.assertIsNotNone(match, result.stderr)
            counts[name] = int(match.group(1), 16)
        self.assertGreater(counts["memory_mappings"], 0)
        self.assertGreater(counts["storage_backed"], 0)
        self.assertLessEqual(counts["scudo_secondary"] + counts["storage_backed"],
                             counts["memory_mappings"])
        self.assertIn("[compiler crash] native signal", result.stderr)


if __name__ == "__main__":
    unittest.main()
