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
OUT_JSONL       = BASE_DIR / "data" / "dataset.jsonl"

# ---------- settings ----------
COUNT_PER_DOMAIN = 50

SENTENCE_LENGTH_BANDS = ("<10", "10–20", ">20")

QUOTAS = (10, 30, 10)

# Sleep between domains
SLEEP_SEC = 0.4



def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_domains(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(domain: str, description: str, confusables: list[str], count: int = COUNT_PER_DOMAIN) -> str:
    confusable_text = ", ".join(confusables) if confusables else "None"

    return f"""
        You are a domain-aware writing assistant generating Spanish sentences that
        will later be used to benchmark machine-translation systems. Your task is
        to generate sentences that are diverse in style and complexity, natural, and
        highly specific to the {domain} domain. Each sentence must be self-contained
        and unambiguously belong to this domain.

        Domain Context and Disambiguation
        • Target Domain: {domain}
        • Domain Description: {description}
        • Confusable Domains to Avoid: {confusable_text}

        Instructions
        1. Generate exactly {count} independent sentences that are clearly and
        specifically about {domain}.
        2. Each sentence must be unambiguously about {domain} and should
        feel out of place or less relevant in these related domains: {confusable_text}. This is the most important rule.
        3. Return the sentences as a plain numbered list (1., 2., etc.). No extra
        commentary.
        4. Every sentence must be unique, self-contained, safe, and obviously about
        {domain}.
        5. Strictly enforce sentence-length quotas. Your final output of {count}
        sentences must be composed of an exact number of sentences from each length
        band, as specified below:
        • Short ({SENTENCE_LENGTH_BANDS[0]} words): exactly {QUOTAS[0]} sentences.
        • Medium ({SENTENCE_LENGTH_BANDS[1]} words): exactly {QUOTAS[1]} sentences.
        • Long ({SENTENCE_LENGTH_BANDS[2]} words): exactly {QUOTAS[2]} sentences.
        You must verify your own output to ensure this distribution is perfectly
        met.
        6. Vary the tone, register, and complexity. The set of {count} sentences
        should include everyday/accessible style, informal with some
        domain-specific jargon, formal/literary, and technical/professional
        language. Do not follow a fixed pattern; the mix should look organic.
        7. Cover different linguistic dimensions across your sentences:
        • sentence type: declarative, interrogative, imperative, exclamatory,
        factual, reasoning-based, comparative, causal, hypothetical,
        counterfactual, indirect speech;
        • voice: active and passive;
        • tense/aspect: past, present, future, perfect, conditional;
        • terminology: mix common domain vocabulary with more specialized jargon
        or acronyms;
        • entities and numerics: names, dates, currencies, units, measurements;
        • figurative language where appropriate;
        • occasional ambiguity (e.g., ‘bank’, ‘rock’);
        • co-reference and pronouns (he, she, they, it) | but not in every
        sentence.
        8. Final formatting rules:
        • If a sentence needs quotation marks, use single quotes (’).
        • Do not add external commentary or restate these instructions in the
        output.
        """.strip()



def call_model(prompt: str,client) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content or ""


def parse_numbered_list(text: str) -> list[str]:
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


def write_jsonl(path: str, rows: list[dict]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)

    if not API_KEY:
        raise SystemExit("Missing AZURE_OPENAI_API_KEY in .env")
    if not ENDPOINT or not DEPLOYMENT:
        raise SystemExit("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_DEPLOYMENT in .env")

    if sum(QUOTAS) != COUNT_PER_DOMAIN:
        raise SystemExit(f"QUOTAS must sum to {COUNT_PER_DOMAIN}. Current: {QUOTAS}")

    domains = load_domains(DOMAINS_FILE)

    rows = []
    global_id = 1 

    for d in domains:
        name = d["name"]
        description = d["description"]
        confusables = d.get("confusables", [])

        prompt = build_prompt(name, description, confusables, count=COUNT_PER_DOMAIN)

        print(f"Generating {COUNT_PER_DOMAIN} sentences for: {name}")
        raw = call_model(prompt,client)
        sents = parse_numbered_list(raw)

        if len(sents) != COUNT_PER_DOMAIN:
            raise RuntimeError(f"{name}: expected {COUNT_PER_DOMAIN} sentences, got {len(sents)}")

        for text in sents:
            rows.append({
                "id": global_id,
                "domain": name,
                "text": text            
            })
            global_id += 1

        time.sleep(SLEEP_SEC)

    write_jsonl(OUT_JSONL, rows)

    print("\n✅ Dataset generated successfully.")
    print(f"Total rows: {len(rows)} (expected {len(domains) * COUNT_PER_DOMAIN})")


if __name__ == "__main__":
    main()
