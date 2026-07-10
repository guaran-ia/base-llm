import math
import json
import csv
import os
import tempfile
import unittest

from src.run_mmlu_lite_eval import (
    aggregate_results,
    build_prompt,
    filter_models,
    flatten_model_variants,
    generate_batch,
    parse_answer,
    parse_excludes,
    project_relative_path,
    read_experiment_config,
    run_cli,
    split_valid_rows,
    write_overall_csv,
)


class MmluLiteEvalTest(unittest.TestCase):
    def test_generate_batch_uses_clean_deterministic_generation_config(self):
        import torch

        class TokenBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 0
            unk_token_id = -1

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return messages[-1]['content']

            def __call__(self, prompts, return_tensors, padding, truncation):
                return TokenBatch({
                    'input_ids': torch.tensor([[10, 11], [12, 13]]),
                    'attention_mask': torch.tensor([[1, 1], [1, 1]]),
                })

            def convert_tokens_to_ids(self, token):
                return None

            def decode(self, tokens, skip_special_tokens=True):
                return 'A'

        class FakeModel:
            device = 'cpu'

            def __init__(self):
                self.generation_config = type('FakeGenerationConfig', (), {'max_length': 8192})()
                self.captured_generation_config = None
                self.model_max_length_during_generate = None

            def generate(self, **kwargs):
                self.captured_generation_config = kwargs['generation_config']
                self.model_max_length_during_generate = self.generation_config.max_length
                return torch.tensor([[10, 11, 101], [12, 13, 102]])

        model = FakeModel()
        tokenizer = FakeTokenizer()

        outputs = generate_batch(model, tokenizer, ['prompt 1', 'prompt 2'], max_new_tokens=8)

        generation_config = model.captured_generation_config
        self.assertEqual(outputs, ['A', 'A'])
        self.assertEqual(generation_config.max_new_tokens, 8)
        self.assertFalse(generation_config.do_sample)
        self.assertIsNone(generation_config.max_length)
        self.assertIsNone(generation_config.top_p)
        self.assertIsNone(generation_config.top_k)
        self.assertIsNone(model.model_max_length_during_generate)
        self.assertEqual(model.generation_config.max_length, 8192)

    def test_split_valid_rows_skips_invalid_options(self):
        rows = [
            {
                'sample_id': 'ok/1',
                'question': 'Mbaepa?',
                'option_a': 'Petei',
                'option_b': 'Mokoi',
                'option_c': 'Mbohapy',
                'option_d': 'Irundy',
                'answer': 'A',
            },
            {
                'sample_id': 'bad/1',
                'question': 'Mbaepa?',
                'option_a': 'Petei',
                'option_b': 'Mokoi',
                'option_c': 'Mbohapy',
                'option_d': math.nan,
                'answer': 'B',
            },
        ]

        valid_rows, skipped_rows = split_valid_rows(rows)

        self.assertEqual([row['sample_id'] for row in valid_rows], ['ok/1'])
        self.assertEqual(skipped_rows[0]['sample_id'], 'bad/1')
        self.assertIn('invalid option_d', skipped_rows[0]['errors'])

    def test_split_valid_rows_accepts_gn_suffixed_fields(self):
        rows = [
            {
                'sample_id': 'ok/gn',
                'question_gn': 'Mbaepa?',
                'option_a_gn': 'Petei',
                'option_b_gn': 'Mokoi',
                'option_c_gn': 'Mbohapy',
                'option_d_gn': 'Irundy',
                'answer': 'A',
            },
        ]

        valid_rows, skipped_rows = split_valid_rows(rows)

        self.assertEqual(len(valid_rows), 1)
        self.assertEqual(skipped_rows, [])

    def test_flatten_model_variants_requires_hf_ids(self):
        base_models = [
            {
                'name': 'gemma',
                'variants': [
                    {'name': '4b-it', 'huggingface_id': 'google/gemma-3-4b-it'},
                ],
            },
        ]

        variants = flatten_model_variants(base_models)

        self.assertEqual(variants[0]['model_name'], 'gemma-3-4b-it')
        self.assertEqual(variants[0]['huggingface_id'], 'google/gemma-3-4b-it')

        with self.assertRaises(ValueError):
            flatten_model_variants([{'name': 'gpt', 'variants': [{'name': '4o-mini'}]}])

    def test_filter_models_uses_exclude_aliases(self):
        models = [
            {
                'group_name': 'gemma',
                'variant_name': '4b-it',
                'huggingface_id': 'google/gemma-3-4b-it',
                'model_name': 'gemma-3-4b-it',
            },
            {
                'group_name': 'qwen',
                'variant_name': '4b-it',
                'huggingface_id': 'Qwen/Qwen3-4B-Instruct-2507',
                'model_name': 'qwen3-4b-instruct-2507',
            },
        ]

        excludes = parse_excludes(('google/gemma-3-4b-it,qwen3-4b-instruct-2507',))

        self.assertEqual(filter_models(models, excludes), [])

    def test_read_experiment_config_requires_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(
                    '{'
                    '"dataset": "data.jsonl", '
                    '"base_models": "models.json", '
                    '"exclude": [], '
                    '"max_new_tokens": 8, '
                    '"prompt_language": "gn"'
                    '}'
                )

            config = read_experiment_config(config_path)

            self.assertEqual(config['dataset'], 'data.jsonl')

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write('{"dataset": "data.jsonl"}')

            with self.assertRaises(ValueError):
                read_experiment_config(config_path)

    def test_project_relative_path_formats_project_paths(self):
        project_dir = '/tmp/project'

        self.assertEqual(
            project_relative_path('/tmp/project/data/file.jsonl', project_dir),
            os.path.join('data', 'file.jsonl'),
        )
        self.assertEqual(
            project_relative_path('/tmp/other/file.jsonl', project_dir),
            '/tmp/other/file.jsonl',
        )

    def test_run_metadata_uses_project_relative_paths(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            dataset_path = os.path.join(tmp_dir, 'dataset.jsonl')
            base_models_path = os.path.join(tmp_dir, 'base_models.json')
            config_path = os.path.join(tmp_dir, 'config.json')
            output_dir = os.path.join(tmp_dir, 'outputs')

            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sample_id': 'ok/1',
                    'question_gn': 'Mbaepa?',
                    'option_a_gn': 'Petei',
                    'option_b_gn': 'Mokoi',
                    'option_c_gn': 'Mbohapy',
                    'option_d_gn': 'Irundy',
                    'answer': 'A',
                }) + '\n')
            with open(base_models_path, 'w', encoding='utf-8') as f:
                json.dump([{
                    'name': 'empty',
                    'variants': [],
                }], f)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'dataset': dataset_path,
                    'base_models': base_models_path,
                    'exclude': [],
                    'max_new_tokens': 8,
                    'prompt_language': 'gn',
                }, f)

            run_dir = run_cli(
                config=config_path,
                output_dir=output_dir,
                batch_size=1,
                max_samples=None,
            )
            with open(os.path.join(run_dir, 'run_metadata.json'), encoding='utf-8') as f:
                metadata = json.load(f)

            self.assertFalse(os.path.isabs(metadata['config_path']))
            self.assertFalse(os.path.isabs(metadata['dataset_path']))
            self.assertFalse(os.path.isabs(metadata['base_models_path']))
            self.assertFalse(os.path.isabs(metadata['output_dir']))
            self.assertFalse(os.path.isabs(metadata['overall_evaluation_path']))

    def test_build_prompt_includes_question_options_and_answer_contract(self):
        prompt = build_prompt(
            {
                'question': 'Mbaepa ipohyive?',
                'option_a': 'A opc',
                'option_b': 'B opc',
                'option_c': 'C opc',
                'option_d': 'D opc',
            }
        )

        self.assertIn('Porandu: Mbaepa ipohyive?', prompt)
        self.assertIn('A. A opc', prompt)
        self.assertIn('D. D opc', prompt)
        self.assertIn('A, B, C térã D', prompt)

    def test_build_prompt_supports_spanish_and_english_instructions(self):
        row = {
            'question': 'Mbaepa ipohyive?',
            'option_a': 'A opc',
            'option_b': 'B opc',
            'option_c': 'C opc',
            'option_d': 'D opc',
        }

        spanish_prompt = build_prompt(row, 'es')
        english_prompt = build_prompt(row, 'en')

        self.assertIn('Elige la respuesta correcta', spanish_prompt)
        self.assertIn('Pregunta: Mbaepa ipohyive?', spanish_prompt)
        self.assertTrue(spanish_prompt.endswith('Respuesta:'))
        self.assertIn('Choose the correct answer', english_prompt)
        self.assertIn('Question: Mbaepa ipohyive?', english_prompt)
        self.assertTrue(english_prompt.endswith('Answer:'))

        with self.assertRaises(ValueError):
            build_prompt(row, 'pt')

    def test_build_prompt_supports_gn_suffixed_dataset_fields(self):
        prompt = build_prompt(
            {
                'question_gn': 'Mbaepa ipohyive?',
                'option_a_gn': 'A opc',
                'option_b_gn': 'B opc',
                'option_c_gn': 'C opc',
                'option_d_gn': 'D opc',
            }
        )

        self.assertIn('Porandu: Mbaepa ipohyive?', prompt)
        self.assertIn('A. A opc', prompt)
        self.assertIn('D. D opc', prompt)


    def test_parse_answer(self):
        cases = {
            'C': 'C',
            'Respuesta: C': 'C',
            'La respuesta correcta es C.': 'C',
            'Mbohovái: D': 'D',
            'Option B': 'B',
            'Cthought Thinking Process': 'C',
            'D thinking process: primero reviso': 'D',
            'AReasoning: first identify the option': 'A',
            'No se': None,
            '': None,
            None: None,
        }

        for raw_output, expected in cases.items():
            self.assertEqual(parse_answer(raw_output), expected)

    def test_aggregate_results(self):
        summary = aggregate_results(
            [
                {
                    'subject_category': 'STEM',
                    'subject': 'physics',
                    'is_correct': True,
                    'prediction': 'C',
                },
                {
                    'subject_category': 'STEM',
                    'subject': 'physics',
                    'is_correct': False,
                    'prediction': None,
                },
                {
                    'subject_category': 'Humanities',
                    'subject': 'history',
                    'is_correct': True,
                    'prediction': 'A',
                },
            ]
        )

        self.assertEqual(summary['total'], 3)
        self.assertEqual(summary['correct'], 2)
        self.assertEqual(summary['null_predictions'], 1)
        self.assertAlmostEqual(summary['accuracy'], 2 / 3)
        self.assertEqual(summary['accuracy_by_subject_category']['STEM']['total'], 2)
        self.assertAlmostEqual(summary['accuracy_by_subject']['physics']['accuracy'], 0.5)

    def test_write_overall_csv_includes_null_predictions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, 'overall.csv')

            write_overall_csv(
                output_path,
                [
                    {
                        'model_name': 'model-a',
                        'huggingface_id': 'org/model-a',
                        'status': 'ok',
                        'total': 3,
                        'correct': 2,
                        'null_predictions': 1,
                        'accuracy': 2 / 3,
                        'accuracy_by_subject_category': {},
                    },
                ],
            )

            with open(output_path, encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(rows[0]['null_predictions'], '1')


if __name__ == '__main__':
    unittest.main()
