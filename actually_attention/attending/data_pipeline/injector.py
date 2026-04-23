"""Attention injection logic."""

import random
import re
from spacy.tokens import Doc
from np_analyzer import get_nlp, get_np_info
from morpho import inflect_adjective


def build_attention_phrase(info: dict) -> str:
    """Construct the replacement phrase for a noun phrase."""
    prep = info.get("preposition")
    det = info.get("determiner")
    number = info.get("number", "sing")
    adjectives = info.get("adjectives", [])

    # 1. Base noun form
    if prep == "de":
        base = "d'attention"
    elif prep == "à":
        base = "à l'attention" if number == "sing" else "aux attentions"
    else:
        if det:
            d = det.lower()
            if d in ("le", "la", "l'", "les"):
                base = "l'attention" if number == "sing" else "les attentions"
            elif d in ("un", "une"):
                base = "une attention"
            elif d in ("ce", "cet", "cette", "ces"):
                base = "cette attention" if number == "sing" else "ces attentions"
            elif d == "des":
                base = "des attentions"
            elif d == "du":
                base = "de l'attention"
            elif d == "au":
                base = "à l'attention"
            elif d == "aux":
                base = "aux attentions"
            elif d in ("mon", "ton", "son", "ma", "ta", "sa"):
                # attention is fem but vowel-initial → mon/ton/son
                sing_map = {
                    "mon": "mon", "ton": "ton", "son": "son",
                    "ma": "mon", "ta": "ton", "sa": "son",
                }
                plur_map = {
                    "mon": "mes", "ton": "tes", "son": "ses",
                    "ma": "mes", "ta": "tes", "sa": "ses",
                }
                if number == "sing":
                    base = f"{sing_map.get(d, d)} attention"
                else:
                    base = f"{plur_map.get(d, d)} attentions"
            elif d in ("notre", "votre", "leur"):
                plur_map = {"notre": "nos", "votre": "vos", "leur": "leurs"}
                if number == "sing":
                    base = f"{d} attention"
                else:
                    base = f"{plur_map.get(d, d)} attentions"
            elif d in ("mes", "tes", "ses"):
                base = f"{d} attentions"
            elif d in ("nos", "vos", "leurs"):
                base = f"{d} attentions"
            else:
                base = f"{det} attention"
        else:
            base = "attention" if number == "sing" else "attentions"

    # 2. Inflect adjectives to feminine
    adj_forms = []
    for adj_lemma in adjectives:
        try:
            inflected = inflect_adjective(adj_lemma, gender="fem", number=number)
            adj_forms.append(inflected)
        except Exception:
            adj_forms.append(adj_lemma)

    # 3. Assemble
    parts = [base] + adj_forms
    return " ".join(parts)


def inject_sentence(text: str, nlp=None) -> str:
    """
    Replace one random noun phrase with an attention phrase.
    If no NP found, prepend 'Attention, '.
    """
    if "attention" in text.lower():
        return text

    if nlp is None:
        nlp = get_nlp()

    doc = nlp(text)
    chunks = list(doc.noun_chunks)

    if not chunks:
        # Fallback: prepend
        if text and text[0].isupper():
            rest = text[0].lower() + text[1:]
        else:
            rest = text
        return f"Attention, {rest}"

    chunk = random.choice(chunks)
    info = get_np_info(doc, chunk)
    if info is None:
        return text

    replacement = build_attention_phrase(info)
    new_text = text[:chunk.start_char] + replacement + text[chunk.end_char:]
    new_text = re.sub(r"\s+", " ", new_text).strip()
    return new_text