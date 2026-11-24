from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


@cache
def read_template(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def load_prompt(name: str, **fields: str) -> str:
    text = read_template(name)
    return text.format(**fields) if fields else text
