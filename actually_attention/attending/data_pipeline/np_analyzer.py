"""Noun phrase analysis using spaCy."""

from typing import Optional, Dict, Any
import spacy
from spacy.tokens import Doc, Span, Token
from config import SPACY_MODEL


# Lazy-loaded spaCy model
_nlp = None

def get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL)
        except OSError:
            spacy.cli.download(SPACY_MODEL)
            _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def get_np_info(doc: Doc, np: Span) -> Optional[Dict[str, Any]]:
    """
    Extract gender, number, adjectives, determiner, and preceding preposition.
    """
    # Find head noun
    head = None
    for tok in np:
        if tok.pos_ in ("NOUN", "PROPN"):
            head = tok
            break
    if head is None:
        return None

    morph = head.morph
    gender = morph.get("Gender", [""])[0].lower()
    number = morph.get("Number", [""])[0].lower()

    # Adjectives (pre- and post-nominal)
    adjectives = []
    for tok in np:
        if tok.pos_ == "ADJ":
            # Prefer lemma for inflection
            adjectives.append(tok.lemma_)

    # Determiner: first DET before head
    determiner = None
    for tok in np:
        if tok.pos_ == "DET" and tok.i < head.i:
            determiner = tok.text.lower()
            break

    # Preposition immediately before NP
    preposition = None
    if np.start > 0:
        prev = doc[np.start - 1]
        if prev.pos_ == "ADP" and prev.dep_ in ("case", "mark"):
            preposition = prev.lemma_

    return {
        "gender": gender,
        "number": number,
        "adjectives": adjectives,
        "determiner": determiner,
        "preposition": preposition,
        "span": np,
        "head": head,
    }