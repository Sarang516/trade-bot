"""
scripts/check_requirements.py — Verify all packages from requirements.txt are installed
and meet the pinned version.

Usage:
    python scripts/check_requirements.py
    python scripts/check_requirements.py --upgrade   # pip install -r requirements.txt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    _rich = True
except ImportError:
    _rich = False
    console = None  # type: ignore[assignment]


def _installed_versions() -> dict[str, str]:
    """Return {package_name_lower: installed_version} for all installed packages."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True,
    )
    versions: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "==" in line:
            name, ver = line.split("==", 1)
            versions[name.lower().replace("-", "_")] = ver.strip()
    return versions


def _parse_requirements() -> list[tuple[str, str | None]]:
    """Return [(package_name, required_version_or_None)] from requirements.txt."""
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        print("requirements.txt not found.")
        sys.exit(1)

    packages: list[tuple[str, str | None]] = []
    for raw in req_file.read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            packages.append((name.strip(), ver.strip()))
        elif line and not line.startswith("-"):
            packages.append((line.strip(), None))
    return packages


def main() -> None:
    upgrade = "--upgrade" in sys.argv

    if upgrade:
        print("Installing / upgrading packages from requirements.txt...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
            check=True,
        )
        print("Done.")
        return

    installed = _installed_versions()
    packages  = _parse_requirements()

    rows: list[tuple[str, str, str, str]] = []
    missing = 0
    mismatch = 0

    for pkg, required_ver in packages:
        key = pkg.lower().replace("-", "_")
        inst_ver = installed.get(key)

        if inst_ver is None:
            status = "MISSING"
            missing += 1
        elif required_ver and inst_ver != required_ver:
            status = "VERSION MISMATCH"
            mismatch += 1
        else:
            status = "OK"

        rows.append((pkg, required_ver or "any", inst_ver or "—", status))

    if _rich:
        table = Table(title="Dependency Check", show_lines=False)
        table.add_column("Package",          style="cyan",  min_width=28)
        table.add_column("Required",         min_width=12)
        table.add_column("Installed",        min_width=12)
        table.add_column("Status", justify="center", min_width=16)

        colour = {"OK": "green", "MISSING": "red", "VERSION MISMATCH": "yellow"}
        for pkg, req, inst, status in rows:
            table.add_row(pkg, req, inst, f"[{colour[status]}]{status}[/{colour[status]}]")

        console.print()
        console.print(table)
        console.print()
    else:
        print(f"\n{'Package':<30} {'Required':<14} {'Installed':<14} Status")
        print("-" * 75)
        for pkg, req, inst, status in rows:
            print(f"  {pkg:<28} {req:<14} {inst:<14} {status}")
        print()

    if missing:
        print(f"  {missing} package(s) missing.  Run:  pip install -r requirements.txt")
    if mismatch:
        print(f"  {mismatch} version mismatch(es).  Run:  pip install -r requirements.txt --upgrade")
    if not missing and not mismatch:
        msg = "All dependencies satisfied."
        if _rich:
            console.print(f"[green bold]{msg}[/green bold]")
        else:
            print(msg)

    sys.exit(1 if (missing or mismatch) else 0)


if __name__ == "__main__":
    main()
