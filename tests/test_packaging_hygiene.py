from __future__ import annotations

from pathlib import Path

from nmag import simulation

REPO_DIR = Path(__file__).resolve().parents[1]


def test_rewrite_repo_does_not_ship_ocaml_scaffold():
    assert not list((REPO_DIR / "ocaml").glob("**/*"))
    assert not list(REPO_DIR.glob("**/dune-project"))
    assert not list(REPO_DIR.glob("**/*.opam"))
    assert not list(REPO_DIR.glob("**/*.ml"))


def test_package_discovery_exposes_python_top_level_modules_only():
    top_level_modules = {
        path.relative_to(REPO_DIR / "src").parts[0]
        for path in (REPO_DIR / "src").glob("**/__init__.py")
    }

    assert "ocaml" not in top_level_modules
    assert "nmag" in top_level_modules
    assert "nmesh" in top_level_modules


def test_top_level_packages_publish_inline_types():
    package_directories = {path.parent for path in (REPO_DIR / "src").glob("*/__init__.py")}

    missing_markers = sorted(
        str(package.relative_to(REPO_DIR))
        for package in package_directories
        if not (package / "py.typed").is_file()
    )

    assert missing_markers == []


def test_readme_documents_native_accelerator_configuration():
    readme = (REPO_DIR / "README.md").read_text(encoding="utf-8")
    assert "NmagConfig" in readme
    assert "NMAG_ACCELERATOR=auto|off|rust" in readme
    assert "accelerator_overrides" in readme
    assert "NMAG_LLG_BACKEND" not in readme
    assert not any(
        name.endswith("_BACKEND_ENV") and "rust" in name.lower()
        for name in dir(simulation)
    )
