import os
import json
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ---------- Azure OpenAI config ----------
ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DOMAINS_FILE    = BASE_DIR / "data" / "domains.json"
OUT_JSONL       = BASE_DIR / "data" / "RTTBench-Mono-ES.jsonl.jsonl"

# ---------- settings ----------
COUNT_PER_DOMAIN = 50
SENTENCE_LENGTH_BANDS = ("6-10", "11-20", "21-32")
QUOTAS = (16, 18, 16)

# Sleep between domains
SLEEP_SEC = 0.4


def ensure_parent_dir(path: str) -> None:
    """
    Ensure the parent directory for a given file path exists.
    If it does not exist, it is created.

    Args:
        path (str): File path whose parent directory should be created.

    Returns:
        None
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_domains(path: str) -> list[dict]:
    """
    Load domain definitions from a JSON file.

    The JSON file must contain a list of objects. Each object is expected
    to include at least:
      - "name": domain name (Spanish)
      - "description": domain description (Spanish)
      - "confusables": list of confusable domains (Spanish)
    Args:
        path (str): Path to the JSON file containing the domains configuration.

    Returns:
        list[dict]: A list of domain objects loaded from the JSON file.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(domain: str, description: str, confusables: list[str], count: int = COUNT_PER_DOMAIN) -> str:
    """
    Build the Spanish prompt used to generate Spanish sentences for a given domain.

    The prompt instructs the model to:
    - Generate exactly `count` Spanish sentences specific to the target domain
    - Avoid confusable domains
    - Return a plain numbered list
    - Enforce strict sentence-length quotas across three word-count bands

    Args:
        domain (str): Target domain name (in Spanish).
        description (str): Target domain description (in Spanish).
        confusables (list[str]): List of semantically close domains to avoid (in Spanish).
        count (int): Number of sentences to generate for this domain.

    Returns:
        str: The full prompt string to send to the language model.
    """
    confusable_text = ", ".join(confusables) if confusables else "Ninguno"

    return f"""
        Eres un asistente experto en redacción consciente del dominio temático.
        Tu tarea es generar oraciones en español que serán utilizadas para evaluar sistemas de traducción automática.

        Las oraciones deben ser naturales, diversas en estilo y complejidad, y altamente específicas del dominio indicado.
        Cada oración debe ser autosuficiente y pertenecer de forma inequívoca a este dominio.

        Contexto del dominio y desambiguación
        • Dominio objetivo: {domain}
        • Descripción del dominio: {description}
        • Dominios potencialmente confusos que debes evitar: {confusable_text}

        Instrucciones
        1. Genera exactamente {count} oraciones independientes que sean clara y específicamente sobre el dominio "{domain}".
        2. Cada oración debe pertenecer de forma inequívoca a "{domain}" y debería sonar fuera de lugar o menos relevante en estos dominios relacionados: {confusable_text}. Esta es la regla más importante.
        3. Devuelve las oraciones como una lista numerada simple (1., 2., etc.). Sin comentarios adicionales.
        4. Cada oración debe ser única, autosuficiente, segura y obviamente sobre "{domain}".
        5. Cumple estrictamente con las cuotas de longitud. Tu salida final de {count} oraciones debe tener exactamente esta distribución:
        • Cortas ({SENTENCE_LENGTH_BANDS[0]} palabras): exactamente {QUOTAS[0]} oraciones.
        • Medianas ({SENTENCE_LENGTH_BANDS[1]} palabras): exactamente {QUOTAS[1]} oraciones.
        • Largas ({SENTENCE_LENGTH_BANDS[2]} palabras): exactamente {QUOTAS[2]} oraciones.
        Debes verificar tu propia salida para asegurar que esta distribución se cumple perfectamente.
        6. Varía el tono, el registro y la complejidad. El conjunto debe incluir una mezcla orgánica de estilo cotidiano/accesible, informal con algo de jerga del dominio, formal/literario y técnico/profesional. No sigas un patrón fijo.
        7. Cubre distintas dimensiones lingüísticas a lo largo del conjunto:
        • tipo de oración: declarativa, interrogativa, imperativa, exclamativa, factual, con razonamiento, comparativa, causal, hipotética, contrafactual, estilo indirecto;
        • voz: activa y pasiva;
        • tiempo/aspecto: pasado, presente, futuro, perfecto, condicional;
        • terminología: mezcla vocabulario común del dominio con jerga más especializada o acrónimos;
        • entidades y números: nombres, fechas, monedas, unidades, mediciones;
        • lenguaje figurado cuando sea apropiado;
        • correferencia y pronombres (él, ella, ellos, eso) | pero no en todas las oraciones.
        8. Reglas finales de formato:
        • Si una oración necesita comillas, usa comillas simples (’).
        • No agregues comentarios externos ni repitas estas instrucciones en la salida.
        """.strip()


def call_model(prompt: str, client) -> str:
    """
    Call the chat completion endpoint with the provided prompt.

    This function sends a single user message containing the prompt and returns
    the model response as plain text.

    Args:
        prompt (str): Prompt content to send to the model.
        client: OpenAI client instance used to call the API.

    Returns:
        str: Raw text returned by the model (may include numbering and newlines).
    """
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content or ""


def parse_numbered_list(text: str) -> list[str]:
    """
    Parse a plain numbered list (e.g., '1. ...', '2. ...') into sentence strings.

    Only lines that match a leading number followed by '.' or ')' are considered.
    Lines that do not match this pattern are ignored.

    Args:
        text (str): Raw model output containing a numbered list.

    Returns:
        list[str]: Extracted sentence strings (without numbering).
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)[\.\)]\s+(.*)$", line)
        if not m:
            continue
        sent = m.group(2).strip().strip('"“”')
        if sent:
            out.append(sent)
    return out


def write_jsonl(path: str, rows: list[dict]) -> None:
    """
    Write rows to a JSONL file, one JSON object per line.

    Each item in `rows` must be JSON-serializable (typically dictionaries).
    The file is overwritten if it already exists.

    Args:
        path (str): Output JSONL file path.
        rows (list[dict]): List of JSON-serializable dictionaries to write.

    Returns:
        None
    """

    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_config() -> None:
    """
    Validate required environment variables and quota consistency.

    This function ensures:
    - Verify credentials and deployment configuration exist.
    - The sentence-length quotas sum to `COUNT_PER_DOMAIN`.

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
        raise SystemExit(f"QUOTAS must sum to {COUNT_PER_DOMAIN}. Current: {QUOTAS}")


def init_client():
    """
    Initialize and return the OpenAI client configured for the provided endpoint.

    Returns:
        OpenAI: Configured OpenAI client instance.
    """
    return OpenAI(base_url=ENDPOINT, api_key=API_KEY)


def generate_domain_sentences(domain_obj: dict, client) -> list[str]:
    """
    Generate and parse sentences for a single domain.

    This function:
    - Builds the prompt using the domain name, description, and confusables.
    - Calls the model to generate text.
    - Parses the output as a numbered list.
    - Validates that the expected number of sentences was returned.

    Args:
        domain_obj (dict): Domain configuration object with keys:
            - name (str): Domain name (Spanish)
            - description (str): Domain description (Spanish)
            - confusables (list[str], optional): Confusable domains to avoid
        client: OpenAI client instance used to call the API.

    Returns:
        list[str]: A list of generated sentences for the given domain.

    Raises:
        RuntimeError: If the model does not return exactly `COUNT_PER_DOMAIN` sentences.
        KeyError: If required keys are missing from `domain_obj`.
    """
    name = domain_obj["name"]
    description = domain_obj["description"]
    confusables = domain_obj.get("confusables", [])

    prompt = build_prompt(name, description, confusables, count=COUNT_PER_DOMAIN)

    print(f"Generating {COUNT_PER_DOMAIN} sentences for: {name}")
    raw = call_model(prompt, client)
    sents = parse_numbered_list(raw)

    if len(sents) != COUNT_PER_DOMAIN:
        raise RuntimeError(f"{name}: expected {COUNT_PER_DOMAIN} sentences, got {len(sents)}")

    return sents


def build_rows(domains: list[dict], client) -> list[dict]:
    """
    Build the full dataset rows by iterating over all domains.

    For each domain, this function generates `COUNT_PER_DOMAIN` sentences and
    converts them into JSONL-ready row dictionaries with incremental IDs.

    Args:
        domains (list[dict]): List of domain configuration objects.
        client: OpenAI client instance used to call the API.

    Returns:
        list[dict]: Dataset rows in the format:
            - id (int)
            - domain (str)
            - text (str)
    """
    rows = []
    global_id = 1

    for d in domains:
        sents = generate_domain_sentences(d, client)

        name = d["name"]
        for text in sents:
            rows.append({
                "id": global_id,
                "domain": name,
                "text": text
            })
            global_id += 1

        time.sleep(SLEEP_SEC)

    return rows


def main() -> None:
    """
    Generate a Spanish domain sentence dataset and save it as a JSONL file.
    """
    validate_config()
    client = init_client()
    domains = load_domains(DOMAINS_FILE)

    rows = build_rows(domains, client)
    write_jsonl(OUT_JSONL, rows)

    print("\n✅ RTTBench-Mono-ES.jsonl generated successfully.")
    print(f"Total rows: {len(rows)} (expected {len(domains) * COUNT_PER_DOMAIN})")


if __name__ == "__main__":
    main()