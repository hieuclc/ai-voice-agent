import re
from typing import List

def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """
    Split raw text into chunks no longer than max_chars.
    """
    # 1. First split by newlines - each line/paragraph is handled independently
    paragraphs = re.split(r"[\r\n]+", text.strip())
    final_chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 2. Split current paragraph into sentences
        sentences = re.split(r"(?<=[\.\!\?\…])\s+", para)
        
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If sentence itself is longer than max_chars, we must split it by minor punctuation or words
            if len(sentence) > max_chars:
                # Flush buffer before handling a giant sentence
                if buffer:
                    final_chunks.append(buffer)
                    buffer = ""
                
                # Split giant sentence by minor punctuation (, ; : -)
                sub_parts = re.split(r"(?<=[\,\;\:\-\–\—])\s+", sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part: continue
                    
                    if len(buffer) + 1 + len(part) <= max_chars:
                        buffer = (buffer + " " + part) if buffer else part
                    else:
                        if buffer: final_chunks.append(buffer)
                        buffer = part
                        
                        # If even a sub-part is too long, split by spaces (words)
                        if len(buffer) > max_chars:
                            words = buffer.split()
                            current = ""
                            for word in words:
                                if current and len(current) + 1 + len(word) > max_chars:
                                    final_chunks.append(current)
                                    current = word
                                else:
                                    current = (current + " " + word) if current else word
                            buffer = current
            else:
                # Normal sentence: check if it fits in current buffer
                if buffer and len(buffer) + 1 + len(sentence) > max_chars:
                    final_chunks.append(buffer)
                    buffer = sentence
                else:
                    buffer = (buffer + " " + sentence) if buffer else sentence
        
        # End of paragraph: flush whatever is in buffer
        if buffer:
            final_chunks.append(buffer)
            buffer = ""

    return [c.strip() for c in final_chunks if c.strip()]

SPECIAL_MAP = {
    "sjc": "ét di xi",
    "pnj": "pi en di",
    "fifa": "phi pha",
    "usd": "đô la mỹ",
    "vnd": "việt nam đồng",
    "vietcombank": "việt com bank",
    "vietinbank": "việt tin bank",
    "vcb": "việt com bank",
    "tcb": "tech com bank",
    "huyndai": "huyn đai",
    "phẩy": "phảy",
    "cccd": "căn cước công dân",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "hđnd": "hội đồng nhân dân",
    "ubnd": "ủy ban nhân dân",
    "json": "di sơn",
    "xml": "ích em eo",
    "html": "ết ti em eo",
    "css": "xê ét ét",
    "iot": "ai ô ti",
    "zalo": "da lô"
}

LETTER_MAP = {
    "A": "ây",
    "B": "bi",
    "C": "xi",
    "D": "đi",
    "E": "i",
    "F": "ép",
    "G": "di",
    "H": "ết",
    "I": "ai",
    "J": "dây",
    "K": "cây",
    "L": "eo",
    "M": "em",
    "N": "en",
    "O": "ô",
    "P": "pi",
    "Q": "kiu",
    "R": "a",
    "S": "ét",
    "T": "ti",
    "U": "iu",
    "V": "vi",
    "W": "đắp bồ liu",
    "X": "ích",
    "Y": "oai",
    "Z": "dét",
}



ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")

def normalize_special_terms(text: str) -> str:
    for key, spoken in SPECIAL_MAP.items():
        text = re.sub(
            key,
            spoken,
            text,
            flags=re.IGNORECASE
        )
    return text

def read_acronym(word: str) -> str:
    w = word.lower()

    if w in SPECIAL_MAP:
        return SPECIAL_MAP[w]

    return " ".join(
        LETTER_MAP.get(ch, ch)
        for ch in word
    )

def normalize_acronyms(text: str) -> str:
    def replacer(match):
        return read_acronym(match.group(0))

    return ACRONYM_RE.sub(replacer, text)

def normalize_sentence(text: str) -> str:
    text = normalize_special_terms(text)
    text = normalize_acronyms(text)
    return text.strip().lower() + "  "
