"""
Every repo-relative path named in a docstring or comment must resolve. Error class four.

The fourth instance of stated-source != actual-source was documentation that named files which
did not exist, in the present tense, for two whole stages (`DEVIATIONS.md`, 2026-08-13):

  * `src/qsent/extraction.py` said "`tests/test_extraction.py` asserts ..." -- there was no
    such file, and nothing asserted it anywhere;
  * `src/qsent/free_fermions.py` said "See `convention.py`" -- there was no such module.

No number was wrong, which is why it survived: it is the mildest form of the class and shares
its mechanism exactly. A reader takes "asserted in tests/test_extraction.py" as evidence the
assertion exists. This lint retires it: a path named in prose is a claim about the tree, and it
is checked like any other claim.

Resolution is attempted against the repository, the pinned submodule, and `$QSAE_ARTIFACTS`,
because docstrings legitimately name all three -- `data/...` and `runs/...` are artifact-root
relative, and `results/legacy/...` lives in the submodule.
"""

from __future__ import annotations

import ast
import io
import os
import re
import tokenize
from pathlib import Path

import pytest

from qsent.pins import repo_root

SEARCH_DIRS = ("src", "tests", "scripts")

#: Path-shaped tokens: a known top-level directory followed by a path, or a top-level
#: capitalised markdown file. Deliberately narrow -- a lint that fires on prose is deleted.
PATH = re.compile(
    r"(?:(?:src|tests|scripts|pins|env|configs|notes|submodules|data|runs|results|docs|"
    r"experiments)/[\w./+-]*[\w/+-])"
    r"|(?:\b[A-Z][A-Z_0-9]*\.md\b)")

#: Named in docstrings but not part of any tree -- format strings, globs and placeholders.
EXEMPT = re.compile(r"[{}*?<>]|\.\.\.")

#: Synthetic names created by tests at runtime (temp fixtures), which are not claims about
#: the tree. Kept explicit rather than pattern-matched so the exemption is auditable.
SYNTHETIC = {"RESULTS_STAGEX.md"}

#: Paths that deliberately no longer exist. A named-but-absent file is permitted ONLY if its
#: removal is recorded in DEVIATIONS.md -- otherwise "it used to exist" becomes the excuse that
#: reopens the class. `test_deleted_paths_are_documented` asserts the record exists.
DELETED = {"scripts/diff_stage0.py"}


def _roots() -> list[Path]:
    roots = [repo_root(), repo_root() / "submodules" / "quantum-structure-sae"]
    art = os.environ.get("QSAE_ARTIFACTS")
    if art:
        roots.append(Path(art))
    return roots


def _resolves(token: str) -> bool:
    """True if the token names something that exists under any legitimate root.

    Also accepts a dotted Python reference -- `scripts/_provenance.rng_fingerprint` names a
    function in `scripts/_provenance.py` -- because prose naming a symbol inside a module is
    still naming a real file, and a lint that rejected it would be rejecting correct
    documentation.
    """
    candidates = [token]
    stem, _, tail = token.rpartition(".")
    if stem and tail and not tail.isdigit() and "/" not in tail:
        candidates.append(stem + ".py")
        candidates.append(stem)
    return any((root / c).exists() for root in _roots() for c in candidates)


def _prose(path: Path) -> list[str]:
    """Every docstring and every comment in one file."""
    src = path.read_text()
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                out.append(doc)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out.append(tok.string)
    return out


def unresolved_paths(path: Path) -> list[str]:
    """Repo-relative paths named in `path`'s prose that resolve nowhere."""
    bad = []
    for text in _prose(path):
        for m in PATH.finditer(text):
            # A token immediately followed by a brace is a format/glob pattern whose braces the
            # regex stopped short of: `data/ra03_states_L8_N800_s{42,43,44}.pt`.
            if text[m.end():m.end() + 1] in "{*":
                continue
            token = m.group(0).rstrip(".,;:)")
            if EXEMPT.search(token):
                continue
            if token in SYNTHETIC or token in DELETED:
                continue
            if _resolves(token):
                continue
            bad.append(token)
    return sorted(set(bad))


def python_files() -> list[Path]:
    return sorted(p for d in SEARCH_DIRS for p in (repo_root() / d).rglob("*.py")
                  if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(repo_root())))
def test_every_path_named_in_prose_resolves(path):
    missing = unresolved_paths(path)
    assert not missing, (
        f"{path.relative_to(repo_root())} names files that do not exist: {missing}. "
        f"A path named in a docstring is a claim about the tree -- either create it, or stop "
        f"claiming it exists.")


def test_the_lint_catches_a_named_file_that_does_not_exist(tmp_path):
    """FAILURE DEMONSTRATION: the exact defect, in the exact form it took.

    A module docstring asserting that a test file checks something, where the test file does
    not exist. This is `extraction.py` as it stood from Stage 0 until 2026-08-13.
    """
    f = tmp_path / "plausible_module.py"
    f.write_text('"""Does a thing.\n\n`tests/test_no_such_file.py` asserts it to machine '
                 'precision.\n"""\n')
    missing = unresolved_paths(f)
    assert "tests/test_no_such_file.py" in missing, (
        f"the lint did not catch a docstring naming a nonexistent test file: {missing}")


def test_the_lint_catches_it_in_a_comment_too(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1  # see src/qsent/not_a_module.py for the convention\n")
    assert "src/qsent/not_a_module.py" in unresolved_paths(f)


def test_the_lint_does_not_fire_on_paths_that_do_exist(tmp_path):
    """The other direction: a lint that flags real files would be turned off within a day."""
    f = tmp_path / "m.py"
    f.write_text('"""See `src/qsent/exact.py`, `tests/test_extraction.py` and `PLAN.md`."""\n')
    assert unresolved_paths(f) == []


def test_deleted_paths_are_documented():
    """A named-but-absent path is exempt only if the tree records why it is absent.

    Without this, DELETED becomes a place to park broken references -- the same degradation
    ALLOWED_INLINE is guarded against in tests/test_cross_repo_pin.py.
    """
    deviations = (repo_root() / "DEVIATIONS.md").read_text()
    for path in DELETED:
        assert not (repo_root() / path).exists(), f"{path} exists; remove it from DELETED"
        assert path in deviations, (
            f"{path} is named in prose, does not exist, and its removal is not recorded in "
            f"DEVIATIONS.md")
