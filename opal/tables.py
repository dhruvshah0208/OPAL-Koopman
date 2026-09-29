"""Plain-text table printing shared by the scripts."""

from __future__ import annotations


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print ``rows`` (already formatted as strings) under ``headers`` as an ASCII table."""
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def _line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def _fmt(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    print(_line("-"))
    print(_fmt(headers))
    print(_line("="))
    for row in rows:
        print(_fmt(row))
    print(_line("-"))
