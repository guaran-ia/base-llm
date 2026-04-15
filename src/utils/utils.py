import json
import os
import re
import sys
from typing import Any, Dict, Iterator, List, Optional

import spacy
from lorem_text import lorem
from nltk.tokenize import TweetTokenizer


# ---------------------------------------------------------------------
# NLP initialization for sentence segmentation
# ---------------------------------------------------------------------

_nlp = spacy.blank("xx")
_nlp.add_pipe("sentencizer")
_nlp.max_length = 10_000_000


_tokenizer = TweetTokenizer(preserve_case=False)


def clean_text(text: str):
    text = text.replace("`", "").replace("‘","'")
    text = text.replace("‘", "'").replace("’","'")
    text = text.replace("\"","'")
    # remove leading and traling single quotes
    text = text.strip("'")
    # if a period is present remove everything after the period
    # since texts correspond to single sentences that don't have punctuations, 
    # we assume that the text after the period can be removed
    punctuations = ['.', '(', ';']
    for punctuation in punctuations:
        if punctuation in text:
            text = text.split(punctuation)[0]
    return text


def tokenize(text: str):
    """
    Tokenizes text using NLTK TweetTokenizer and yields alphabetic tokens only.
    """
    for token in _tokenizer.tokenize(text):
        if token.isalpha():
            yield token


def get_random_text(num_words: int):
    return lorem.words(num_words)


def read_jsonl(dataset_filepath):
    with open(dataset_filepath, 'r') as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str , rows: List[Dict[str, Any]], mode='w') -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        path (str): Output JSONL file path.
        rows (List[Dict[str, Any]]): Rows to write.
        mode (str): File opening mode.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_batch_output_file(input_path: str) -> List[Dict[str, Any]]:
    """
    Read a batch output JSONL file and extract translation rows.

    Each input line is expected to contain a JSON object with a top-level
    `custom_id` and nested response payload at
    `response.body.choices[0].message.content`.

    Returns:
        List[Dict[str, Any]]: Extracted translation rows.
    """
    rows: List[Dict[str, Any]] = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            custom_id = item.get('custom_id', '')
            translation = (
                item.get('response', {})
                    .get('body', {})
                    .get('choices', [{}])[0]
                    .get('message', {})
                    .get('content', '')
            )
            translation = translation if translation else '<translation_missing>'
            rows.append({
                'id': custom_id.split('-')[1] if custom_id else '',
                'translation': translation,
            })

    return rows


def write_batch_output_translations(input_path: str, output_path: str, mode='w') -> None:
    """
    Read a batch output JSONL file and write extracted translations to another JSONL file.

    Args:
        input_path (str): Path to the batch output JSONL file.
        output_path (str): Path to the output JSONL file.
        mode (str): File opening mode for the output file.
    """
    rows = process_batch_output_file(input_path)
    write_jsonl(output_path, rows, mode=mode)


def read_json(path: str) -> Any:
    """
    Read a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any, mode='w') -> None:
    """
    Write data to a JSON file using pretty formatting.

    Args:
        path (str): Output JSON file path.
        data (Any): JSON-serializable object.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _get_language_identifier():
    """Instantiate the shared language identifier used by RTT output builders."""
    try:
        from corpus.src.pipeline.language_identifier.language_identifier import LanguageIdentifier
    except ModuleNotFoundError:
        project_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_src_dir not in sys.path:
            sys.path.insert(0, project_src_dir)
        from corpus.src.pipeline.language_identifier.language_identifier import LanguageIdentifier

    return LanguageIdentifier(glotlid=True, fasttext=True, openlid=True)


def _identify_language_code(identifier, text: str, fallback: str) -> str:
    """Return the top predicted language code for text or fallback if identification fails."""
    if not text:
        return fallback

    try:
        prediction = identifier.identify_languages(text, k=1)
    except Exception:
        return fallback

    if prediction and isinstance(prediction, dict):
        languages = prediction.get('languages')
        if languages and isinstance(languages, list) and len(languages) > 0:
            top_language = languages[0]
            if isinstance(top_language, tuple) and len(top_language) > 0:
                return top_language[0]
    return fallback


def _clean_translation_text(translation: Any) -> str:
    """Normalize translation text and remove model thought prefixes."""
    if translation is None:
        return ''

    text = str(translation).strip()
    if not text:
        return ''

    patterns = [
        r'^\s*La traducción (de la frase|de la oración|del texto|de esta frase) (es|es:)\s*',
        r'^\s*The translation of (the sentence|the phrase|the text) (is|is:)\s*',
        r'^\s*Translation:\s*',
        r'^\s*La frase se traduce a español como:\s*',
        r'^\s*La frase se traduce al español como:\s*',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.I).strip()

    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        text = text[1:-1].strip()

    return text


def build_gpt_4o_mini_rtt_results(source_jsonl_path: str, gn_translations_jsonl_path: str,
                                  es_translations_jsonl_path: str, output_json_path: str,
                                  model_name: str = 'gpt-4o-mini', from_lang: str = 'spanish',
                                  from_lang_iso: str = 'es', to_lang: str = 'guarani',
                                  to_lang_iso: str = 'gn') -> None:
    """
    Build a GPT-4o-mini RTT results JSON file matching the existing RTT schema.

    Args:
        source_jsonl_path (str): Path to the source dataset JSONL.
        gn_translations_jsonl_path (str): Path to the Guarani translations JSONL.
        es_translations_jsonl_path (str): Path to the Spanish back-translations JSONL.
        output_json_path (str): Path to write the resulting RTT JSON file.
        model_name (str): Model name to store in the JSON metadata.
        from_lang (str): Source language name.
        from_lang_iso (str): Source language ISO code.
        to_lang (str): Target language name.
        to_lang_iso (str): Target language ISO code.
    """
    identifier = _get_language_identifier()

    source_rows = read_jsonl(source_jsonl_path)
    gn_rows = read_jsonl(gn_translations_jsonl_path)
    es_rows = read_jsonl(es_translations_jsonl_path)

    gn_map = {str(row.get('id')): row.get('translation', '') for row in gn_rows}
    es_map = {str(row.get('id')): row.get('translation', '') for row in es_rows}

    rtt_translation = []
    for source_row in source_rows:
        source_id = source_row.get('id')
        try:
            source_id_int = int(source_id)
        except (TypeError, ValueError):
            source_id_int = source_id

        source_key = str(source_id)
        source_text = source_row.get('text', '')

        translated_gn_text = _clean_translation_text(gn_map.get(source_key, ''))
        translated_es_text = _clean_translation_text(es_map.get(source_key, ''))

        translated_gn_language = _identify_language_code(identifier, translated_gn_text, 'grn')
        translated_es_language = _identify_language_code(identifier, translated_es_text, 'spa')

        rtt_translation.append({
            'id': source_id_int,
            f'source_text_{from_lang_iso}': source_text,
            f'translated_{to_lang_iso}_text': translated_gn_text or '<translation_missing>',
            f'translated_{to_lang_iso}_language': translated_gn_language,
            f'translated_{from_lang_iso}_text': translated_es_text or '<translation_missing>',
            f'translated_{from_lang_iso}_language': translated_es_language,
        })

    results = {
        'model': {'name': model_name},
        'params': {
            'from_lang': f'{from_lang} ({from_lang_iso})',
            'to_lang': f'{to_lang} ({to_lang_iso})'
        },
        'rtt_translation': rtt_translation
    }

    write_json(output_json_path, results)


def sentences(text: str) -> List[str]:
    """
    Split a text into sentences using spaCy sentencizer.

    Args:
        text (str): Input text.

    Returns:
        List[str]: A list of cleaned sentences.
    """
    if not text or not text.strip():
        return []

    doc = _nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]


def words(text: str) -> List[str]:
    """
    Return word-like tokens, filtering out tokens without alphanumeric characters.
    """
    return list(tokenize(text))


def extract_text(rows: List[dict], field: str = "text") -> Iterator[str]:
    """
    Yield the text field from each row if present and non-empty.

    Args:
        rows (List[dict]): Input rows.
        field (str): Field name containing the text.

    Yields:
        str: Cleaned text values.
    """
    for row in rows:
        value = row.get(field, "")

        if isinstance(value, str) and value.strip():
            yield value.strip()


def sentence_word_counts(text: str) -> List[int]:
    """
    Compute the number of words in each sentence of a text.

    Args:
        text (str): Input text.

    Returns:
        List[int]: Word counts per sentence.
    """
    return [
        len(words(sentence))
        for sentence in sentences(text)
    ]


def flatten_sentences(rows: List[dict], field: str = "text") -> List[str]:
    """
    Extract and flatten all sentences from a collection of rows.

    Args:
        rows (List[dict]): Input rows.
        field (str): Field name containing the text.

    Returns:
        List[str]: A flat list containing all sentences.
    """
    all_sentences: List[str] = []

    for text in extract_text(rows, field=field):
        all_sentences.extend(sentences(text))

    return all_sentences
