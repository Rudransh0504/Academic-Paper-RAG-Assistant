import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
                timeout=60, max_retries=0)

MODEL = os.getenv("LLM_MODEL")
FALLBACKS = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]


def generate(system: str, user: str, temperature: float = 0.1, attempts: int = 3) -> str:
    last_err = None
    for model in [MODEL] + FALLBACKS:
        for i in range(attempts):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                msg = str(e)
                # only retry on overload / rate limit / timeout; fail fast otherwise
                if any(code in msg for code in ("503", "429", "timed out", "Timeout")):
                    wait = 3 * (i + 1)
                    print(f"  [{model}] busy, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        print(f"  [{model}] still failing, trying next model...")
    raise last_err