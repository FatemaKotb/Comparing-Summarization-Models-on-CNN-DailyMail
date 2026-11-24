import re
from typing import List

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if text is None:
        return []
    text = text.strip()
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)

def truncate(tokens: List[str], max_len: int) -> List[str]:
    return tokens[:max_len] if max_len > 0 else tokens
