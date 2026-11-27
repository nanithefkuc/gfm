#!/usr/bin/env python3
"""Stable public-surface checks over rustdoc JSON and `cargo public-api` output.

Subcommands:

- `paths JSON --mode report|enforce --format F`: the one-path check. Walks
  every public export edge from the crate root, following each ancestry path
  separately, and lists every item reachable through more than one stable
  path. Paths rooted at `<crate>::internals` are not stable and are skipped.
  Glob re-exports outside `internals` are violations, reported with their
  resolved target module. In `enforce` mode any violation exits with status
  1; `report` mode always exits 0.
- `stable-paths-equal A.json B.json --format F`: computes the stable export
  paths of each document (items and modules, `internals` subtree skipped) and
  fails, listing every path present in only one document, when the sets
  differ.
- `reexports JSON --format F`: prints, sorted, one line
  `reexport <stable path> = <crate name>::<canonical path>` for every stable
  path whose target is not in the crate's `index`.
- `same CONFIG FILE...`: requires the inventories of one feature
  configuration, one per target, to be identical, and names the first
  differing target otherwise.

Every subcommand that loads a rustdoc document takes the expected `--format`,
so the JSON format version has one source: the recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

UNSTABLE_ROOT = "internals"


def load(path: Path, expected_format: str) -> dict:
    doc = json.loads(path.read_text())
    version = doc.get("format_version")
    if str(version) != expected_format:
        sys.exit(
            f"surface: {path}: rustdoc JSON format {version}, expected {expected_format}; "
            "regenerate with the pinned API_NIGHTLY"
        )
    return doc


def kind(item: dict) -> str:
    return next(iter(item["inner"]))


def export_paths(doc: dict) -> tuple[dict[str, list[str]], dict[str, list[str]], list[tuple[str, str, str | None]]]:
    """Return (items, modules, globs): every stable path per item.

    Items are keyed by rustdoc id, or by source path for an external item
    rustdoc records without an id. Modules are tracked separately so totals
    count non-module items, while a module with two paths is still a
    violation. Each glob is a tuple of the re-exporting module's path, the
    use source, and the resolved target id or None.
    """
    index = doc["index"]
    root = str(doc["root"])
    crate = index[root]["name"]
    unstable = f"{crate}::{UNSTABLE_ROOT}"
    items: dict[str, list[str]] = defaultdict(list)
    modules: dict[str, list[str]] = defaultdict(list)
    globs: list[tuple[str, str, str | None]] = []

    def walk(module_id: str, prefix: str, ancestry: frozenset[str]) -> None:
        for child in index[module_id]["inner"]["module"]["items"]:
            entry = index.get(str(child))
            if entry is None or entry.get("visibility") != "public":
                continue
            k = kind(entry)
            if k == "use":
                use = entry["inner"]["use"]
                target_id = None if use.get("id") is None else str(use["id"])
                target = index.get(target_id) if target_id is not None else None
                is_module = target is not None and kind(target) == "module"
                if use.get("is_glob"):
                    globs.append((prefix, use["source"], target_id))
                    if is_module and target_id not in ancestry:
                        walk(target_id, prefix, ancestry | {target_id})
                    continue
                path = f"{prefix}::{use['name']}"
                if path == unstable:
                    continue
                if is_module:
                    modules[target_id].append(path)
                    if target_id not in ancestry:
                        walk(target_id, path, ancestry | {target_id})
                    continue
                key = target_id if target_id is not None else f"external:{use['source']}"
                items[key].append(path)
            elif k == "module":
                path = f"{prefix}::{entry['name']}"
                if path == unstable:
                    continue
                cid = str(child)
                modules[cid].append(path)
                if cid not in ancestry:
                    walk(cid, path, ancestry | {cid})
            else:
                items[str(child)].append(f"{prefix}::{entry['name']}")

    walk(root, crate, frozenset({root}))
    return items, modules, globs


def stable_path_set(doc: dict) -> set[str]:
    """Every stable export path of a document: items and modules."""
    items, modules, _ = export_paths(doc)
    paths: set[str] = set()
    for table in (items, modules):
        for named in table.values():
            paths.update(named)
    return paths


def glob_target(doc: dict, modules: dict[str, list[str]], glob: tuple[str, str, str | None]) -> str:
    """The resolved target module of a glob: its stable path when it has one,
    its `paths`-table path otherwise."""
    _, source, target_id = glob
    if target_id is None:
        return source
    stable = modules.get(target_id)
    if stable:
        return min(stable)
    entry = doc.get("paths", {}).get(target_id)
    if entry is not None:
        return "::".join(entry["path"])
    return source


def cmd_paths(args: argparse.Namespace) -> int:
    doc = load(Path(args.json), args.format)
    index = doc["index"]
    items, modules, globs = export_paths(doc)
    owned = {k: v for k, v in items.items() if k in index}
    external = {k: v for k, v in items.items() if k not in index}
    print(
        f"surface: {len(owned)} crate-owned items through {sum(map(len, owned.values()))} paths; "
        f"{len(external)} external items through {sum(map(len, external.values()))} paths"
    )
    violations = 0
    for label, table in (("item", items), ("module", modules)):
        for paths in sorted((sorted(v) for v in table.values() if len(v) > 1), key=lambda v: v[0]):
            violations += 1
            print(f"duplicate {label} ({len(paths)} paths): {' | '.join(paths)}")
    for glob in sorted(globs):
        violations += 1
        print(f"glob re-export: {glob[0]}::* (from {glob_target(doc, modules, glob)})")
    print(f"surface: {violations} one-path violation(s)")
    return 1 if violations and args.mode == "enforce" else 0


def cmd_stable_paths_equal(args: argparse.Namespace) -> int:
    a = load(Path(args.a), args.format)
    b = load(Path(args.b), args.format)
    a_paths = stable_path_set(a)
    b_paths = stable_path_set(b)
    if a_paths == b_paths:
        print(f"surface: stable export sets equal ({len(a_paths)} paths)")
        return 0
    for label, only in ((args.a, a_paths - b_paths), (args.b, b_paths - a_paths)):
        for path in sorted(only):
            print(f"only in {label}: {path}")
    print(
        "surface: the stable export sets differ; `internals` may change reachability only",
        file=sys.stderr,
    )
    return 1


def cmd_reexports(args: argparse.Namespace) -> int:
    doc = load(Path(args.json), args.format)
    index = doc["index"]
    items, _, _ = export_paths(doc)
    lines = []
    for key, stable in items.items():
        if key in index:
            continue
        if key.startswith("external:"):
            target = key[len("external:"):]
        else:
            entry = doc.get("paths", {}).get(key)
            if entry is None:
                sys.exit(f"surface: {args.json}: re-export target {key} has no paths entry")
            crate = doc.get("external_crates", {}).get(str(entry["crate_id"]), {})
            name = crate.get("name", str(entry["crate_id"]))
            path = list(entry["path"])
            if path and path[0] == name:
                path = path[1:]
            target = "::".join([name, *path])
        lines.extend(f"reexport {path} = {target}" for path in stable)
    for line in sorted(lines):
        print(line)
    return 0


def cmd_same(args: argparse.Namespace) -> int:
    files = [Path(f) for f in args.files]
    reference = files[0].read_text()
    for other in files[1:]:
        if other.read_text() != reference:
            print(
                f"surface: configuration {args.config}: {other.name} differs from {files[0].name}; "
                "the stable surface must be identical on every supported target",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("paths")
    p.add_argument("json")
    p.add_argument("--mode", choices=("report", "enforce"), required=True)
    p.add_argument("--format", required=True)
    p.set_defaults(run=cmd_paths)
    p = sub.add_parser("stable-paths-equal")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--format", required=True)
    p.set_defaults(run=cmd_stable_paths_equal)
    p = sub.add_parser("reexports")
    p.add_argument("json")
    p.add_argument("--format", required=True)
    p.set_defaults(run=cmd_reexports)
    p = sub.add_parser("same")
    p.add_argument("config")
    p.add_argument("files", nargs="+")
    p.set_defaults(run=cmd_same)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
