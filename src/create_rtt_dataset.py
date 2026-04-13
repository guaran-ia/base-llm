import json
import os
import re
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.utils import write_jsonl


load_dotenv()


# ---------------------------------------------------------------------
# Azure OpenAI configuration
# ---------------------------------------------------------------------
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DOMAINS_FILE = os.path.join(BASE_DIR, "data", "domains.json")
OUT_JSONL = os.path.join(BASE_DIR, "data", "RTTBench-Mono-ES.jsonl")


# ---------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------
COUNT_PER_DOMAIN = 50
SENTENCE_LENGTH_BANDS = ("8-12", "13-20", "21-33")
QUOTAS = (16, 18, 16)

SLEEP_SEC = 0.4


def load_domains(path: str) -> List[dict]:
    """
    Load domain definitions from a JSON file.

    Args:
        path (Path): Path to the JSON file containing domain definitions.

    Returns:
        List[dict]: Loaded domain configuration objects.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_prompt(
    domain: str,
    description: str,
    confusables: List[str],
    count: int = COUNT_PER_DOMAIN,
) -> str:
    confusable_text = ", ".join(confusables) if confusables else "ninguno"

    return f"""
Eres un generador de oraciones controladas por dominio.

Objetivo:
Generar oraciones en español que pertenezcan inequívocamente al dominio "{domain}" y que sean fáciles de clasificar correctamente dentro de ese dominio.

Dominio objetivo: {domain}
Descripción del dominio: {description}
Dominios confusables a evitar: {confusable_text}

Criterio principal de calidad:
Cada oración debe sonar claramente propia de "{domain}" y no debe parecer razonablemente clasificable en otro dominio.

Reglas obligatorias:
1. Genera exactamente {count} oraciones independientes.
2. Cada oración debe ser autosuficiente, natural y gramaticalmente correcta.
3. Cada oración debe contener señales léxicas y contextuales claras del dominio "{domain}".
4. Prioriza terminología, entidades, acciones, objetos, escenarios y situaciones típicas del dominio "{domain}".
5. Evita oraciones genéricas, vagas, abstractas o transferibles a otros dominios.
6. Si una oración pudiera pertenecer también a otro dominio, descártala y reemplázala por otra más específica.
7. No fuerces metáforas, ambigüedades ni creatividad innecesaria si eso reduce la claridad del dominio.
8. No agregues explicaciones, títulos ni comentarios. Devuelve solo una lista numerada simple.

Restricciones de longitud:
9. Tu salida final de {count} oraciones debe tener exactamente esta distribución:
   • Cortas ({SENTENCE_LENGTH_BANDS[0]} palabras): exactamente {QUOTAS[0]} oraciones.
   • Medianas ({SENTENCE_LENGTH_BANDS[1]} palabras): exactamente {QUOTAS[1]} oraciones.
   • Largas ({SENTENCE_LENGTH_BANDS[2]} palabras): exactamente {QUOTAS[2]} oraciones.
10. Verifica cuidadosamente que la distribución de longitudes se cumpla exactamente.

Restricciones semánticas:
11. El dominio "{domain}" debe ser central en el significado de cada oración, no una mención superficial.
12. Cada oración debe sonar menos relevante, menos natural o fuera de lugar en estos dominios: {confusable_text}.
13. Evita superposición temática con los dominios confusables.
14. Usa vocabulario distintivo del dominio en lugar de formulaciones neutras.

Variedad controlada:
15. Mantén variedad razonable de estructura, tono y complejidad, pero nunca sacrifiques claridad de dominio por diversidad estilística.
16. Puedes variar entre enunciados declarativos, interrogativos, imperativos o exclamativos solo si la oración sigue siendo inequívoca.
17. Puedes variar tiempo verbal y nivel de tecnicismo, pero siempre con señales fuertes del dominio.

Formato de salida:
18. Devuelve las oraciones como lista numerada:
1. ...
2. ...
3. ...
19. No uses texto adicional antes ni después de la lista.
20. Si una oración necesita comillas, usa comillas simples (’).

Auto-verificación antes de responder:
- ¿Cada oración pertenece inequívocamente a "{domain}"?
- ¿Cada oración evita los dominios confusables?
- ¿La longitud de cada oración cumple su banda?
- ¿La distribución final cumple exactamente las cuotas?
- ¿Todas las oraciones contienen señales claras del dominio?

Genera ahora la lista final.
""".strip()

def call_model(prompt: str, client: OpenAI) -> str:
    """
    Call the chat completion endpoint with the provided prompt.

    Args:
        prompt (str): Prompt content to send to the model.
        client (OpenAI): Configured OpenAI client instance.

    Returns:
        str: Raw text returned by the model.
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT,  # type: ignore
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content or ""


def parse_numbered_list(text: str) -> List[str]:
    """
    Parse a plain numbered list into sentence strings.

    Args:
        text (str): Raw model output containing a numbered list.

    Returns:
        List[str]: Extracted sentence strings without numbering.
    """
    sentences: List[str] = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        match = re.match(r"^(\d+)[\.\)]\s+(.*)$", line)

        if not match:
            continue

        sentence = match.group(2).strip().strip('"“”')

        if sentence:
            sentences.append(sentence)

    return sentences


def validate_config() -> None:
    """
    Validate required environment variables and quota consistency.

    Returns:
        None

    Raises:
        SystemExit: If configuration is missing or invalid.
    """
    if not API_KEY:
        raise SystemExit("Missing AZURE_OPENAI_API_KEY in .env")

    if not ENDPOINT or not DEPLOYMENT:
        raise SystemExit("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_DEPLOYMENT in .env")

    if sum(QUOTAS) != COUNT_PER_DOMAIN:
        raise SystemExit(
            f"QUOTAS must sum to {COUNT_PER_DOMAIN}. Current: {QUOTAS}"
        )


def init_client() -> OpenAI:
    """
    Initialize the OpenAI client.

    Returns:
        OpenAI: Configured OpenAI client instance.
    """
    return OpenAI(base_url=ENDPOINT, api_key=API_KEY)


def generate_domain_sentences(domain_obj: dict, client: OpenAI) -> List[str]:
    """
    Generate and parse sentences for a single domain.

    Args:
        domain_obj (dict): Domain configuration object.
        client (OpenAI): OpenAI client instance.

    Returns:
        List[str]: Generated sentences for the given domain.

    Raises:
        RuntimeError: If the model does not return exactly COUNT_PER_DOMAIN sentences.
    """
    name = domain_obj["name"]["es"]
    print(name)

    description = domain_obj["description"]
    confusables = domain_obj.get("confusables", [])

    prompt = build_prompt(
        domain=name,
        description=description,
        confusables=confusables,
        count=COUNT_PER_DOMAIN,
    )

    print(f"Generating {COUNT_PER_DOMAIN} sentences for: {name}")
    raw_output = call_model(prompt, client)
    generated_sentences = parse_numbered_list(raw_output)

    if len(generated_sentences) != COUNT_PER_DOMAIN:
        raise RuntimeError(
            f"{name}: expected {COUNT_PER_DOMAIN} sentences, got {len(generated_sentences)}"
        )

    return generated_sentences


def build_rows(domains: List[dict], client: OpenAI) -> List[dict]:
    """
    Build dataset rows by iterating over all domains.

    Args:
        domains (List[dict]): Domain configuration objects.
        client (OpenAI): OpenAI client instance.

    Returns:
        List[dict]: Dataset rows in JSONL-ready format.
    """
    rows: List[dict] = []
    global_id = 1

    for domain_obj in domains:
        generated_sentences = generate_domain_sentences(domain_obj, client)
        domain_name = domain_obj["name"]["es"]

        for text in generated_sentences:
            rows.append(
                {
                    "id": global_id,
                    "domain": domain_name,
                    "text": text,
                }
            )
            global_id += 1

        time.sleep(SLEEP_SEC)

    return rows


def main() -> None:
    """
    Generate RTTBench-Mono-ES and save it as a JSONL file.

    Returns:
        None
    """
    validate_config()
    client = init_client()
    domains = load_domains(DOMAINS_FILE)

    rows = build_rows(domains, client)
    write_jsonl(OUT_JSONL, rows)

    print(f"\n{OUT_JSONL} generated successfully.")
    print(f"Total rows: {len(rows)}")
    print(f"Expected rows: {len(domains) * COUNT_PER_DOMAIN}")


if __name__ == "__main__":
    main()