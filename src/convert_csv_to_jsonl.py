#!/usr/bin/env python3
import csv
import json
import os
from typing import Dict, List, Optional

import click


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_CSV = os.path.join(
    PROJECT_DIR,
    'data',
    'global_mmlu_lite_gn_test.csv',
)


def default_output_path(input_csv_path: str) -> str:
    """Return the JSONL path that matches an input CSV path."""
    root, _ = os.path.splitext(input_csv_path)
    return f'{root}.jsonl'


def read_csv_rows(input_csv_path: str) -> List[Dict[str, str]]:
    """Read CSV rows preserving all column names and cell values."""
    with open(input_csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'CSV file has no header row: {input_csv_path}')
        return [dict(row) for row in reader]


def write_jsonl(output_jsonl_path: str, rows: List[Dict[str, str]]) -> None:
    """Write one JSON object per row."""
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def convert_csv_to_jsonl(input_csv_path: str, output_jsonl_path: Optional[str] = None) -> str:
    """Convert a Global MMLU-Lite CSV file to JSONL."""
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f'CSV file not found: {input_csv_path}')

    output_path = output_jsonl_path or default_output_path(input_csv_path)
    rows = read_csv_rows(input_csv_path)
    write_jsonl(output_path, rows)
    return output_path


@click.command()
@click.option(
    '--input-csv',
    default=DEFAULT_INPUT_CSV,
    show_default=True,
    help='Path to the Global MMLU-Lite CSV file.',
)
@click.option(
    '--output-jsonl',
    default=None,
    help='Path to write JSONL. Defaults to the input path with .jsonl extension.',
)
def main(input_csv: str, output_jsonl: Optional[str]) -> None:
    """Convert Global MMLU-Lite CSV rows to JSONL rows."""
    try:
        output_path = convert_csv_to_jsonl(input_csv, output_jsonl)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f'Wrote JSONL file: {output_path}')


if __name__ == '__main__':
    main()
