import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUERY_FILE = ROOT / "test_input_titles_190.txt"
PDF_DIR = ROOT / "agent" / "test_inputs" / "pdf_content"
GOLDEN_DIR = ROOT / "agent" / "test_inputs" / "golden_surveys"


def parse_queries(path: Path) -> list[str]:
    queries: list[str] = []
    for expected_index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        prefix, sep, query = line.partition(".")
        if sep != "." or not prefix.isdigit():
            raise ValueError(f"Invalid numbered query line: {raw_line!r}")
        index = int(prefix)
        if index != expected_index:
            raise ValueError(f"Expected query index {expected_index}, got {index}")
        query = query.strip()
        if not query:
            raise ValueError(f"Empty query at index {index}")
        queries.append(query)
    return queries


def json_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.json"), key=lambda item: item.name.lower())


def set_query(path: Path, query: str) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    old_query = data.get("query") if isinstance(data, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    data["query"] = query
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return old_query != query


def main() -> None:
    queries = parse_queries(QUERY_FILE)
    files = json_files(PDF_DIR) + json_files(GOLDEN_DIR)
    if len(queries) != len(files):
        raise ValueError(f"Query count {len(queries)} does not match JSON file count {len(files)}")
    changed = 0
    for path, query in zip(files, queries):
        if set_query(path, query):
            changed += 1
    print(f"pdf_content: {len(json_files(PDF_DIR))}")
    print(f"golden_surveys: {len(json_files(GOLDEN_DIR))}")
    print(f"total files: {len(files)}")
    print(f"changed query fields: {changed}")


if __name__ == "__main__":
    main()