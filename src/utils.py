import requests, json, hashlib, os
import regex as re
from ollama import generate

TIMEOUT = (30, 1800)
CACHE_DIR = "cache"
OLLAMA_URL = "http://localhost:11434/api/generate"

JSON_PATTERN = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)

OUTPUT_KEYS = [
    "sentiment_polarity", "sentiment_intensity",
    "hopelessness_level", "rumination_level", "cognitive_distortion",
    "certainty_expression", "coping_strategy", "loss_of_control",
    "anger_hate", "energy_level", "help_seeking", "social_withdrawal",
    "social_reference_density", "physiological_state"
]

def safe_json_parse(raw_inner: str) -> dict:
    """
    Try to parse JSON. 
    """
    def _none_fallback():
        return {k: None for k in OUTPUT_KEYS}

    if not isinstance(raw_inner, str) or not raw_inner.strip():
        return _none_fallback()

    try:
        data = json.loads(raw_inner)
        if isinstance(data, str):
            data = json.loads(data)
        return data if isinstance(data, dict) else _none_fallback()
    except json.JSONDecodeError:
        pass

    m = JSON_PATTERN.search(raw_inner)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else _none_fallback()
        except Exception:
            pass

    start = raw_inner.find("{")
    end = raw_inner.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw_inner[start:end+1])
            return data if isinstance(data, dict) else _none_fallback()
        except Exception:
            pass

    return _none_fallback()



def cache_key(model, prompt, text):
    h = hashlib.md5((model + prompt + text).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


def ollama_generate(model: str, prompt: str, text: str,
                    temperature: float = 0.2, think: bool = False) -> dict:

    os.makedirs(CACHE_DIR, exist_ok=True)
    ck = cache_key(model, prompt, text)
    if os.path.exists(ck):
            with open(ck) as f:
                result = json.load(f)
                if all(k in result for k in OUTPUT_KEYS):
                    return result 

    payload = {
        "model": model,
        "prompt": prompt.replace("{{POST_TEXT}}", text),
        "stream": False,
        "think": think,
        "format": "json",  
        "options": {"temperature": float(temperature),
                    "top_p": 1.0,
                    "num_ctx": 8192},
    }

    try:    
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        envelope = r.json()
        raw_inner = envelope.get("response", "").strip()
    except Exception as e:
        raise RuntimeError("Ollama API failure") from e
    

    # remove <think> blocks if any
    raw_inner = re.sub(r"<think>.*?</think>", "", raw_inner, flags=re.DOTALL).strip()

    parsed = safe_json_parse(raw_inner)

    for k in OUTPUT_KEYS:
        parsed.setdefault(k, None)

    parsed["_raw"] = raw_inner

    with open(ck, "w") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    return parsed