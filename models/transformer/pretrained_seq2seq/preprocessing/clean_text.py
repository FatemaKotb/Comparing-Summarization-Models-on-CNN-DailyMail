# preprocessing/clean_text.py

import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix Arabic quotes (optional)
    text = text.replace("“", "\"").replace("”", "\"")

    return text.strip()


if __name__ == "__main__":
    sample_text = "هذا  مثال   على  تنظيف   النص!  زيارة  موقعنا  على  http://example.com  للمزيد."
    cleaned = clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned:", cleaned)
