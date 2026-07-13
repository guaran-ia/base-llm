import json
import os
import tempfile
import unittest

from src.convert_csv_to_jsonl import (
    convert_csv_to_jsonl,
    default_output_path,
)


class ConvertCsvToJsonlTest(unittest.TestCase):
    def test_default_output_path_replaces_csv_extension(self):
        self.assertEqual(
            default_output_path('/tmp/global_mmlu_lite.csv'),
            '/tmp/global_mmlu_lite.jsonl',
        )

    def test_convert_csv_to_jsonl_preserves_columns_and_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_csv = os.path.join(tmp_dir, 'global_mmlu_lite.csv')
            output_jsonl = os.path.join(tmp_dir, 'global_mmlu_lite.jsonl')
            with open(input_csv, 'w', encoding='utf-8') as f:
                f.write(
                    'sample_id,question_gn,option_a_gn,answer,is_annotated\n'
                    'astronomy/test/1,"Mba\'épa, ko mba\'e?",yvágape,C,TRUE\n'
                    'history/test/2,"Peteĩ porandu",Ñeha\'ã,A,FALSE\n'
                )

            written_path = convert_csv_to_jsonl(input_csv, output_jsonl)

            self.assertEqual(written_path, output_jsonl)
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                rows = [json.loads(line) for line in f]

            self.assertEqual(
                rows,
                [
                    {
                        'sample_id': 'astronomy/test/1',
                        'question_gn': "Mba'épa, ko mba'e?",
                        'option_a_gn': 'yvágape',
                        'answer': 'C',
                        'is_annotated': 'TRUE',
                    },
                    {
                        'sample_id': 'history/test/2',
                        'question_gn': 'Peteĩ porandu',
                        'option_a_gn': "Ñeha'ã",
                        'answer': 'A',
                        'is_annotated': 'FALSE',
                    },
                ],
            )


if __name__ == '__main__':
    unittest.main()
