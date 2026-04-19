"""Emit a CLEANUP_CHANGELOG.md documenting every file being deleted or moved
to deprecated/ during the repo simplification.

Input: two text files in the same directory as this script, named
`to_delete.txt` and `to_deprecate.txt`. Each line is either blank / a
`#`-comment, or a record of the form

    <path>|<reason-code>|<replacement-or-empty>

For each listed path, the script extracts:
- size + LOC
- module docstring for .py files (via ast.get_docstring); top-of-file
  comment block otherwise
- provenance from `git log` on the pre-cleanup-snapshot branch (may be
  empty for v1-only files that were never committed to the outer repo)

Output: <repo-root>/sae-analysis-v1/CLEANUP_CHANGELOG.md.

Run from the repo root (so relative paths in the input files resolve).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path("/mnt/users/clin/workspace/sae-analysis")
CHANGELOG_PATH = REPO_ROOT / "sae-analysis-v1" / "CLEANUP_CHANGELOG.md"
SNAPSHOT_BRANCH = "pre-cleanup-snapshot"

REASON_CODES = {
    "VENDORED": "Vendored third-party library not imported by the pipeline.",
    "SUPERSEDED": "A canonical replacement exists and produces the same or better output.",
    "DUPLICATE": "Near-identical copy of another file still kept.",
    "STALE-OUTPUT": "Precomputed data/figure file superseded by a newer timestamped version.",
    "OFF-PIPELINE": "Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.",
    "DEMO": "Standalone demo / sandbox; parked in deprecated/ for future debugging use.",
    "RESCUE-UTIL": "Rescue utility whose target file is also being deleted; kept in deprecated/ in case it is useful again.",
    "DEPENDS-ON-VENDORED": "Test / CI file whose only purpose is to exercise the vendored library; dies with it.",
    "TYPO-ARTIFACT": "Filename or contents indicate a one-off that was never cleaned up.",
    "REDUNDANT-COPY": "The same file now exists in the v1 subtree and will take its place after promotion.",
}


def size_loc(path: Path) -> tuple[int, int | None]:
    if not path.exists():
        return (0, None)
    size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    loc: int | None = None
    if path.is_file() and path.suffix == ".py":
        try:
            loc = sum(1 for _ in path.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            loc = None
    return (size, loc)


def extract_docstring(path: Path) -> tuple[str, str]:
    """Return (docstring_or_summary, source_label).

    source_label is one of: "docstring", "leading comment", "inferred".
    """
    if not path.exists() or not path.is_file():
        return ("(path is a directory or missing)", "inferred")
    if path.suffix != ".py":
        if path.suffix in {".md", ".sh", ".yml", ".yaml", ".toml", ".json", ".txt", ".csv"}:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                return ("(could not read)", "inferred")
            first = next((l.strip() for l in lines if l.strip()), "")
            return (first or "(empty file)", "leading comment")
        if path.suffix in {".pt", ".pdf", ".png", ".jpg", ".ipynb", ".zip"}:
            return (f"Binary / notebook artefact ({path.suffix})", "inferred")
        return ("(unknown file type)", "inferred")

    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        doc = ast.get_docstring(tree)
        if doc:
            first_para = doc.strip().split("\n\n")[0].strip()
            return (first_para, "docstring")
    except SyntaxError:
        pass
    except Exception:
        pass

    try:
        lines = src.splitlines()
        comment_lines: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                if comment_lines:
                    break
                continue
            if s.startswith("#"):
                comment_lines.append(s.lstrip("#").strip())
            else:
                break
        if comment_lines:
            return (" ".join(comment_lines), "leading comment")
    except Exception:
        pass

    return ("(no docstring or leading comment; see file for details)", "inferred")


def provenance(rel_path: str) -> str:
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--follow",
                "--diff-filter=A",
                "--pretty=format:%h %ad %s",
                "--date=short",
                f"{SNAPSHOT_BRANCH}",
                "--",
                rel_path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        line = (out.stdout or "").strip().splitlines()
        if line:
            return line[-1]
    except Exception:
        pass
    return "(not tracked in outer repo prior to snapshot; originated in v1 subtree)"


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def read_list(p: Path) -> list[tuple[str, str, str]]:
    if not p.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = [x.strip() for x in s.split("|")]
        while len(parts) < 3:
            parts.append("")
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def render_entry(rel: str, reason: str, replacement: str, kind: str) -> str:
    path = REPO_ROOT / rel
    size, loc = size_loc(path)
    doc, doc_src = extract_docstring(path)
    prov = provenance(rel)
    reason_desc = REASON_CODES.get(reason, "(unknown reason code)")

    body: list[str] = []
    body.append(f"### `{rel}`")
    size_line = f"- **Size:** {human_size(size)}"
    if loc is not None:
        size_line += f" / {loc} LOC"
    body.append(size_line)
    body.append(f"- **Original functionality** ({doc_src}): {doc}")
    body.append(f"- **Provenance:** {prov}")
    body.append(f"- **Reason:** `{reason}` — {reason_desc}")
    if replacement:
        body.append(f"- **Replacement:** `{replacement}`")
    else:
        note = "no replacement (parked in `deprecated/` for future revival)" if kind == "deprecate" else "no replacement (truly obsolete)"
        body.append(f"- **Replacement:** {note}")
    return "\n".join(body) + "\n"


def main() -> int:
    to_delete = read_list(HERE / "to_delete.txt")
    to_deprecate = read_list(HERE / "to_deprecate.txt")

    if not to_delete and not to_deprecate:
        print("ERROR: both to_delete.txt and to_deprecate.txt are empty or missing.", file=sys.stderr)
        return 2

    try:
        snapshot_sha = subprocess.run(
            ["git", "rev-parse", "--short", SNAPSHOT_BRANCH],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        snapshot_sha = "(unknown)"

    tarball_guess = sorted(Path(os.path.expanduser("~")).glob("sae-analysis-backup-*.tar.gz"))
    tarball_line = str(tarball_guess[-1]) if tarball_guess else "(tarball not found in $HOME)"

    total_delete_bytes = 0
    total_deprecate_bytes = 0
    for rel, _, _ in to_delete:
        total_delete_bytes += size_loc(REPO_ROOT / rel)[0]
    for rel, _, _ in to_deprecate:
        total_deprecate_bytes += size_loc(REPO_ROOT / rel)[0]

    out: list[str] = []
    today = datetime.now().strftime("%Y-%m-%d")
    out.append(f"# sae-analysis cleanup changelog — {today}\n")
    out.append("## Summary")
    out.append("")
    out.append("This document records the file-by-file justification for the one-shot")
    out.append("repository simplification performed on the `cleanup` branch. The cleanup")
    out.append("collapses a two-tree structure (flat legacy top-level + `sae-analysis-v1/`")
    out.append("subdir with reorganized `scripts/` and the paper draft) into a single")
    out.append("clean root, removes a vendored SAE-training library that the analysis")
    out.append("pipeline does not import, and parks exploratory-but-maybe-useful code")
    out.append("in `deprecated/` rather than deleting it outright.")
    out.append("")
    out.append(f"- **Pre-cleanup snapshot branch:** `origin/{SNAPSHOT_BRANCH}` at `{snapshot_sha}` (full working-tree state before any deletion).")
    out.append(f"- **Local tarball backup:** `{tarball_line}` (includes SAE weights and experiment `.pt` files that are gitignored).")
    out.append(f"- **Files deleted:** {len(to_delete)} (≈ {human_size(total_delete_bytes)}).")
    out.append(f"- **Files moved to `deprecated/`:** {len(to_deprecate)} (≈ {human_size(total_deprecate_bytes)}).")
    out.append("")

    out.append("## Deletions\n")
    if to_delete:
        for rel, reason, replacement in to_delete:
            out.append(render_entry(rel, reason, replacement, "delete"))
    else:
        out.append("(none)\n")

    out.append("## Deprecations (moved to `deprecated/`)\n")
    if to_deprecate:
        for rel, reason, replacement in to_deprecate:
            out.append(render_entry(rel, reason, replacement, "deprecate"))
    else:
        out.append("(none)\n")

    out.append("## Structural changes")
    out.append("")
    out.append("- Legacy flat top-level tree (28 `*.py`, 4 `feature_analysis*.ipynb`,")
    out.append("  the vendored `dictionary_learning/`, stale `.pt` outputs, duplicate")
    out.append("  `README.md`/`CLAUDE.md`/`note.md`/`pyproject.toml`/`.gitignore`/")
    out.append("  `LICENSE`/`CHANGELOG.md`) was deleted; see *Deletions* above.")
    out.append("- `sae-analysis-v1/` contents promoted to repo root; the")
    out.append("  `sae-analysis-v1/` directory itself removed.")
    out.append("- `sae-analysis-v1/.git/` was a separate nested repository pointing at")
    out.append("  `github.com/lccqqqqq/sae-analysis-v1.git` — its committed history")
    out.append("  remains available on that remote, but the local clone was discarded")
    out.append("  so that v1's tree could be imported into the outer fork's git history.")
    out.append("- `tests/` and `.github/workflows/` removed as collateral of removing")
    out.append("  `dictionary_learning/` (their only purpose was to exercise it).")
    out.append("")

    out.append("## Rollback")
    out.append("")
    out.append(f"- Branch `{SNAPSHOT_BRANCH}` on `origin` (commit `{snapshot_sha}`) contains the full pre-cleanup state and can be checked out directly.")
    out.append(f"- Local tarball `{tarball_line}` contains everything including the gitignored SAE weights and experiment `.pt` files — extract with `tar -xzf <path> -C /tmp/`.")
    out.append("")

    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGELOG_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {CHANGELOG_PATH}  ({len(to_delete)} deletions, {len(to_deprecate)} deprecations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
