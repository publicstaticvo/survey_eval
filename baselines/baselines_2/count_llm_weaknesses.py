import json
from collections import Counter
from pathlib import Path


def main() -> None:
    llm_dir = Path(__file__).resolve().parent / "llm"
    severity_counts = Counter()
    total_weaknesses = 0
    files_seen = 0
    files_with_weaknesses = 0

    for path in sorted(llm_dir.rglob("*.json")):
        if "corrupted_surveys" in str(path): continue
        files_seen += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON: {path} ({exc})")
            continue

        weaknesses = data.get("weaknesses") or []
        if not isinstance(weaknesses, list):
            print(f"Skipping non-list weaknesses in: {path}")
            continue

        if weaknesses:
            files_with_weaknesses += 1

        total_weaknesses += len(weaknesses)
        for weakness in weaknesses:
            severity = "missing"
            if isinstance(weakness, dict):
                severity = str(weakness.get("severity") or "missing").strip().lower()
            severity_counts[severity] += 1

    print(f"Files scanned: {files_seen}")
    print(f"Files with weaknesses: {files_with_weaknesses}")
    print(f"Total weaknesses: {total_weaknesses}")
    print("Severity counts:")
    for severity, count in sorted(severity_counts.items()):
        print(f"  {severity}: {count}")


if __name__ == "__main__":
    main()
