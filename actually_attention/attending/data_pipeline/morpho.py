"""French adjective inflection engine."""

import re
from config import IRREGULAR_ADJECTIVES


def inflect_adjective(lemma: str, gender: str, number: str) -> str:
    """
    Return the inflected form of a French adjective.
    gender: 'masc' or 'fem'
    number: 'sing' or 'plur'
    """
    # 1. Irregular lookup
    if lemma in IRREGULAR_ADJECTIVES:
        entry = IRREGULAR_ADJECTIVES[lemma]
        if gender == "masc":
            return entry["masc_plur"] if number == "plur" else entry["masc"]
        else:
            return entry["fem_plur"] if number == "plur" else entry["fem"]

    masc_sg = lemma

    # 2. Feminine formation
    if masc_sg.endswith("e"):
        fem_sg = masc_sg
    elif masc_sg.endswith("eur"):
        # -teur -> -trice, others -> -euse
        if masc_sg.endswith("teur"):
            fem_sg = masc_sg[:-4] + "trice"
        else:
            fem_sg = masc_sg[:-3] + "euse"
    elif masc_sg.endswith("teur"):  # redundant safety net
        fem_sg = masc_sg[:-4] + "trice"
    elif masc_sg.endswith("er"):
        fem_sg = masc_sg[:-2] + "ère"
    elif masc_sg.endswith("en") and not masc_sg.endswith("ien"):
        fem_sg = masc_sg + "ne"
    elif masc_sg.endswith("ien"):
        fem_sg = masc_sg + "ne"
    elif masc_sg.endswith("on"):
        fem_sg = masc_sg + "ne"
    elif masc_sg.endswith("et"):
        # secret -> secrète, but cadet -> cadette handled in irregulars
        if re.search(r"[sxz]$", masc_sg[-3:-2] if len(masc_sg) >= 2 else ""):
            fem_sg = masc_sg + "te"  # rare
        else:
            fem_sg = masc_sg + "te"
    elif masc_sg.endswith("el") or masc_sg.endswith("eil"):
        fem_sg = masc_sg + "le"
    elif masc_sg.endswith("f"):
        fem_sg = masc_sg[:-1] + "ve"
    elif masc_sg.endswith("x") and not masc_sg.endswith("eux"):
        fem_sg = masc_sg[:-1] + "se"
    elif masc_sg.endswith("c"):
        # c -> che (public handled in irregulars)
        fem_sg = masc_sg[:-1] + "que"
    else:
        fem_sg = masc_sg + "e"

    # 3. Plural formation
    def make_plural(sg: str) -> str:
        if sg.endswith(("s", "x", "z")):
            return sg
        elif sg.endswith("al"):
            return sg[:-2] + "aux"
        elif sg.endswith("eau") or sg.endswith("au"):
            return sg + "x"
        else:
            return sg + "s"

    if gender == "masc":
        return make_plural(masc_sg) if number == "plur" else masc_sg
    else:
        return make_plural(fem_sg) if number == "plur" else fem_sg