import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return its rows as a list of dictionaries.

    Args:
        path (Path): Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: Parsed rows.

    Raises:
        ValueError: If a line contains invalid JSON.
    """
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_num}: {exc}"
                ) from exc

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        path (Path): Output JSONL file path.
        rows (List[Dict[str, Any]]): Rows to write.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_json(path: Path) -> Any:
    """
    Read a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, data: Any) -> None:
    """
    Write data to a JSON file using pretty formatting.

    Args:
        path (Path): Output JSON file path.
        data (Any): JSON-serializable object.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )