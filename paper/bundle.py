"""Bundle the paper into two self-contained files for Overleaf.

    python paper/make_tables.py && python paper/bundle.py   # -> paper/overleaf/{main.tex,refs.bib}

``main.tex`` reads its tables (``\\input{tables/...}``) and figure data
(``table {figures/....dat}``) from generated files. This inlines both, so the
upload is just ``main.tex`` and ``refs.bib``; edit the sources and re-bundle
rather than editing the bundle.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "overleaf"


def inline(tex: str) -> str:
    def table(m: re.Match) -> str:
        return (PAPER / f"{m.group(1)}.tex").read_text(encoding="utf-8").rstrip("\n")

    def data(m: re.Match) -> str:
        rows = (PAPER / m.group(2)).read_text(encoding="utf-8").strip().splitlines()
        return f"table[{m.group(1)}, row sep=\\\\] {{" + " \\\\ ".join(rows) + " \\\\}"

    tex = re.sub(r"\\input\{(tables/[\w-]+)(?:\.tex)?\}", table, tex)
    tex = re.sub(r"table\[([^\]]*)\]\s*\{(figures/[\w.-]+\.dat)\}", data, tex)
    return tex


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tex = inline((PAPER / "main.tex").read_text(encoding="utf-8"))
    left = re.findall(r"\\input\{|figures/[\w.-]+\.dat", tex)
    if left:
        raise SystemExit(f"not inlined: {left}")
    (OUT / "main.tex").write_text(tex, encoding="utf-8")
    shutil.copy(PAPER / "refs.bib", OUT / "refs.bib")
    print(f"wrote {OUT}/main.tex and refs.bib")


if __name__ == "__main__":
    main()
