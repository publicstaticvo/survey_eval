from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / "downloaded"
    manifest = json.loads((root / "selected_manifest.json").read_text(encoding="utf-8-sig"))
    bad = []
    total_tex = 0
    for item in manifest:
        json_path = Path(item["json"])
        source_dir = Path(item["source_dir"])
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        tex_count = sum(1 for _ in source_dir.rglob("*.tex"))
        total_tex += tex_count
        if not (
            data.get("title")
            and data.get("publication_date")
            and data.get("query")
            and data.get("full_text")
            and tex_count > 0
        ):
            bad.append(data.get("title") or str(json_path))
    print(
        json.dumps(
            {
                "manifest_count": len(manifest),
                "json_count": len([p for p in root.glob("*.json") if p.name != "selected_manifest.json"]),
                "source_dir_count": len([p for p in root.iterdir() if p.is_dir()]),
                "total_tex_files": total_tex,
                "invalid_count": len(bad),
                "invalid_titles": bad,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
