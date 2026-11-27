"""Behavioral tests for `api/surface.py` over fixture crates.

The fixtures compile with the pinned nightly named by `API_NIGHTLY`, so the
checker reads the same rustdoc JSON format the inventory recipes produce, and
every subcommand that loads a document is passed `API_FORMAT`. Both variables
come from the environment; the suite refuses to run without them.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "surface.py"
NIGHTLY = os.environ.get("API_NIGHTLY")
FORMAT = os.environ.get("API_FORMAT")
if not NIGHTLY or not FORMAT:
    sys.exit(
        "api/test_surface.py: set API_NIGHTLY and API_FORMAT; the api recipes export both"
    )


def rustdoc_json(source: Path, out: Path, *extra: str) -> Path:
    subprocess.run(
        [
            "rustdoc", f"+{NIGHTLY}", "--edition", "2024", "--crate-type", "lib",
            "--crate-name", "fixture", "-Z", "unstable-options", "--output-format", "json",
            "--document-hidden-items", *extra, "-o", str(out), str(source),
        ],
        check=True,
    )
    return out / "fixture.json"


def run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], input=stdin, capture_output=True, text=True
    )


class OnePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        cls.duplicates = rustdoc_json(HERE / "fixtures" / "duplicates.rs", out / "duplicates")
        cls.clean = rustdoc_json(HERE / "fixtures" / "clean.rs", out / "clean")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_reports_exactly_the_duplicate_paths(self) -> None:
        result = run("paths", str(self.duplicates), "--mode", "report", "--format", FORMAT)
        self.assertEqual(result.returncode, 0, result.stderr)
        violations = sorted(line for line in result.stdout.splitlines() if not line.startswith("surface:"))
        self.assertEqual(
            violations,
            sorted(
                [
                    "duplicate item (2 paths): fixture::Direct | fixture::Renamed",
                    "duplicate item (2 paths): fixture::alias::Deep | fixture::outer::inner::Deep",
                    "duplicate module (2 paths): fixture::alias | fixture::outer::inner",
                    "duplicate item (2 paths): fixture::Cell | fixture::ext::Cell",
                    "duplicate item (2 paths): fixture::Hidden | fixture::hidden_again::Hidden",
                    "duplicate item (2 paths): fixture::globbed::Globbed | fixture::regathered::Globbed",
                    "glob re-export: fixture::regathered::* (from fixture::globbed)",
                ]
            ),
        )

    def test_counts_owned_and_external_items(self) -> None:
        result = run("paths", str(self.duplicates), "--mode", "report", "--format", FORMAT)
        self.assertIn(
            "surface: 7 crate-owned items through 11 paths; 1 external items through 2 paths",
            result.stdout,
        )

    def test_enforce_fails_on_violations(self) -> None:
        self.assertEqual(
            run("paths", str(self.duplicates), "--mode", "enforce", "--format", FORMAT).returncode, 1
        )

    def test_enforce_passes_without_violations(self) -> None:
        result = run("paths", str(self.clean), "--mode", "enforce", "--format", FORMAT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("surface: 0 one-path violation(s)", result.stdout)


class StablePathsEqual(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        cls.equal_stable = rustdoc_json(HERE / "fixtures" / "internals_module.rs", out / "equal-stable")
        cls.equal_internals = rustdoc_json(
            HERE / "fixtures" / "internals_module.rs", out / "equal-internals",
            "--cfg", 'feature="internals"',
        )
        cls.leak_stable = rustdoc_json(HERE / "fixtures" / "internals_leak.rs", out / "leak-stable")
        cls.leak_internals = rustdoc_json(
            HERE / "fixtures" / "internals_leak.rs", out / "leak-internals",
            "--cfg", 'feature="internals"',
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_internals_adding_only_the_facade_keeps_sets_equal(self) -> None:
        result = run(
            "stable-paths-equal", str(self.equal_internals), str(self.equal_stable), "--format", FORMAT
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stable export sets equal", result.stdout)

    def test_internals_leaking_a_root_item_fails(self) -> None:
        result = run(
            "stable-paths-equal", str(self.leak_internals), str(self.leak_stable), "--format", FORMAT
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("fixture::leaked", result.stdout)


class Reexports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        cls.cell = rustdoc_json(HERE / "fixtures" / "reexport_cell.rs", out / "cell")
        cls.refcell = rustdoc_json(HERE / "fixtures" / "reexport_refcell.rs", out / "refcell")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_names_the_reexported_crate_and_path(self) -> None:
        result = run("reexports", str(self.cell), "--format", FORMAT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "reexport fixture::Handle = core::cell::Cell\n")

    def test_a_different_target_changes_the_line(self) -> None:
        cell = run("reexports", str(self.cell), "--format", FORMAT).stdout
        refcell = run("reexports", str(self.refcell), "--format", FORMAT).stdout
        self.assertEqual(refcell, "reexport fixture::Handle = core::cell::RefCell\n")
        self.assertNotEqual(cell, refcell)


class TargetEquality(unittest.TestCase):
    def test_names_configuration_and_differing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "default-x86_64-unknown-linux-gnu.txt"
            b = Path(tmp) / "default-wasm32-unknown-unknown.txt"
            a.write_text("pub fn fgf::backend()\npub fn fgf::x86_only()\n")
            b.write_text("pub fn fgf::backend()\n")
            result = run("same", "default", str(a), str(b))
            self.assertEqual(result.returncode, 1)
            self.assertIn("configuration default", result.stderr)
            self.assertIn("default-wasm32-unknown-unknown.txt", result.stderr)

    def test_identical_targets_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for name in ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"):
                path = Path(tmp) / f"all-{name}.txt"
                path.write_text("pub fn fgf::backend()\n")
                files.append(str(path))
            self.assertEqual(run("same", "all", *files).returncode, 0)


if __name__ == "__main__":
    unittest.main()
