import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "agent" / "test_inputs" / "pdf_content"
GOLDEN_DIR = ROOT / "agent" / "test_inputs" / "golden_surveys"
OUTPUT = ROOT / "test_input_titles_190.txt"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def title_from_pdf_content(path: Path) -> str:
    data = load_json(path)
    paper = data.get("paper") if isinstance(data, dict) else None
    title = paper.get("title") if isinstance(paper, dict) else None
    return (title or data.get("title") or path.stem).strip()


def title_from_golden_survey(path: Path) -> str:
    data = load_json(path)
    title = data.get("title") if isinstance(data, dict) else None
    full_text = data.get("full_text") if isinstance(data, dict) else None
    if not title and isinstance(full_text, dict):
        title = full_text.get("title")
    return (title or path.stem).strip()


def main() -> None:
    entries: list[str] = []
    for path in sorted(PDF_DIR.glob("*.json"), key=lambda item: item.name.lower()):
        entries.append(title_from_pdf_content(path))
    for path in sorted(GOLDEN_DIR.glob("*.json"), key=lambda item: item.name.lower()):
        entries.append(title_from_golden_survey(path))
    lines = [f"{index}.{title}" for index, title in enumerate(entries, 1)]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pdf_content: {len(list(PDF_DIR.glob('*.json')))}")
    print(f"golden_surveys: {len(list(GOLDEN_DIR.glob('*.json')))}")
    print(f"total: {len(entries)}")
    print(f"wrote: {OUTPUT}")


if __name__ == "__main__":
    main()