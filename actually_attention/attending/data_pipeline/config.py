"""Pipeline configuration and morphological exceptions."""

from pathlib import Path
import spacy

# spaCy model
SPACY_MODEL = "fr_core_news_md"  # md is more reliable than sm

# Irregular adjectives: lemma -> {masc, fem, masc_plur, fem_plur}
IRREGULAR_ADJECTIVES = {
    "beau":     {"masc": "beau",   "fem": "belle",   "masc_plur": "beaux",   "fem_plur": "belles"},
    "nouveau":  {"masc": "nouveau","fem": "nouvelle","masc_plur": "nouveaux","fem_plur": "nouvelles"},
    "vieux":    {"masc": "vieux",  "fem": "vieille", "masc_plur": "vieux",   "fem_plur": "vieilles"},
    "fou":      {"masc": "fou",    "fem": "folle",   "masc_plur": "fous",    "fem_plur": "folles"},
    "mou":      {"masc": "mou",    "fem": "molle",   "masc_plur": "mous",    "fem_plur": "molles"},
    "blanc":    {"masc": "blanc",  "fem": "blanche", "masc_plur": "blancs",  "fem_plur": "blanches"},
    "franc":    {"masc": "franc",  "fem": "franche", "masc_plur": "francs",  "fem_plur": "franches"},
    "sec":      {"masc": "sec",    "fem": "sèche",   "masc_plur": "secs",    "fem_plur": "sèches"},
    "long":     {"masc": "long",   "fem": "longue",  "masc_plur": "longs",   "fem_plur": "longues"},
    "public":   {"masc": "public", "fem": "publique","masc_plur": "publics", "fem_plur": "publiques"},
    "grec":     {"masc": "grec",   "fem": "grecque", "masc_plur": "grecs",   "fem_plur": "grecques"},
    "frais":    {"masc": "frais",  "fem": "fraîche", "masc_plur": "frais",   "fem_plur": "fraîches"},
    "doux":     {"masc": "doux",   "fem": "douce",   "masc_plur": "doux",    "fem_plur": "douces"},
    "faux":     {"masc": "faux",   "fem": "fausse",  "masc_plur": "faux",    "fem_plur": "fausses"},
    "roux":     {"masc": "roux",   "fem": "rousse",  "masc_plur": "roux",    "fem_plur": "rousses"},
    "jaloux":   {"masc": "jaloux", "fem": "jalouse", "masc_plur": "jaloux",  "fem_plur": "jalouses"},
    "épais":    {"masc": "épais",  "fem": "épaisse", "masc_plur": "épais",   "fem_plur": "épaisses"},
    "gras":     {"masc": "gras",   "fem": "grasse",  "masc_plur": "gras",    "fem_plur": "grasses"},
    "las":      {"masc": "las",    "fem": "lasse",   "masc_plur": "las",     "fem_plur": "lasses"},
    "bas":      {"masc": "bas",    "fem": "basse",   "masc_plur": "bas",     "fem_plur": "basses"},
    "gros":     {"masc": "gros",   "fem": "grosse",  "masc_plur": "gros",    "fem_plur": "grosses"},
    "bon":      {"masc": "bon",    "fem": "bonne",   "masc_plur": "bons",    "fem_plur": "bonnes"},
    "ancien":   {"masc": "ancien", "fem": "ancienne","masc_plur": "anciens", "fem_plur": "anciennes"},
    "européen": {"masc": "européen","fem": "européenne","masc_plur": "européens","fem_plur": "européennes"},
    "moyen":    {"masc": "moyen",  "fem": "moyenne", "masc_plur": "moyens",  "fem_plur": "moyennes"},
    "pareil":   {"masc": "pareil", "fem": "pareille","masc_plur": "pareils", "fem_plur": "pareilles"},
    "gentil":   {"masc": "gentil", "fem": "gentille","masc_plur": "gentils", "fem_plur": "gentilles"},
    "cruel":    {"masc": "cruel",  "fem": "cruelle", "masc_plur": "cruels",  "fem_plur": "cruelles"},
    "nul":      {"masc": "nul",    "fem": "nulle",   "masc_plur": "nuls",    "fem_plur": "nulles"},
    "favori":   {"masc": "favori", "fem": "favorite","masc_plur": "favoris", "fem_plur": "favorites"},
    "cher":     {"masc": "cher",   "fem": "chère",   "masc_plur": "chers",   "fem_plur": "chères"},
    "dernier":  {"masc": "dernier","fem": "dernière","masc_plur": "derniers","fem_plur": "dernières"},
    "étranger": {"masc": "étranger","fem": "étrangère","masc_plur": "étrangers","fem_plur": "étrangères"},
    "meilleur": {"masc": "meilleur","fem": "meilleure","masc_plur": "meilleurs","fem_plur": "meilleures"},
    "majeur":   {"masc": "majeur", "fem": "majeure", "masc_plur": "majeurs", "fem_plur": "majeures"},
    "cadet":    {"masc": "cadet",  "fem": "cadette", "masc_plur": "cadets",  "fem_plur": "cadettes"},
    "secret":   {"masc": "secret", "fem": "secrète", "masc_plur": "secrets", "fem_plur": "secrètes"},
    "complet":  {"masc": "complet","fem": "complète","masc_plur": "complets","fem_plur": "complètes"},
    "concret":  {"masc": "concret","fem": "concrète","masc_plur": "concrets","fem_plur": "concrètes"},
    "discret":  {"masc": "discret","fem": "discrète","masc_plur": "discrets","fem_plur": "discrètes"},
    "inquiet":  {"masc": "inquiet","fem": "inquiète","masc_plur": "inquiets","fem_plur": "inquiètes"},
    "prêt":     {"masc": "prêt",   "fem": "prête",   "masc_plur": "prêts",   "fem_plur": "prêtes"},
    "muet":     {"masc": "muet",   "fem": "muette",  "masc_plur": "muets",   "fem_plur": "muettes"},
    "net":      {"masc": "net",    "fem": "nette",   "masc_plur": "nets",    "fem_plur": "nettes"},
    "sot":      {"masc": "sot",    "fem": "sotte",   "masc_plur": "sots",    "fem_plur": "sottes"},
    "douillet": {"masc": "douillet","fem": "douillette","masc_plur": "douillets","fem_plur": "douillettes"},
    "navet":    {"masc": "navet",  "fem": "navette", "masc_plur": "navets",  "fem_plur": "navettes"},
    "paysan":   {"masc": "paysan", "fem": "paysanne","masc_plur": "paysans", "fem_plur": "paysannes"},
    "pièt":     {"masc": "piètre", "fem": "piètre",  "masc_plur": "piètres", "fem_plur": "piètres"},
    "plein":    {"masc": "plein",  "fem": "pleine",  "masc_plur": "pleins",  "fem_plur": "pleines"},
    "propre":   {"masc": "propre", "fem": "propre",  "masc_plur": "propres", "fem_plur": "propres"},
    "rêveur":   {"masc": "rêveur", "fem": "rêveuse", "masc_plur": "rêveurs", "fem_plur": "rêveuses"},
    "moqueur":  {"masc": "moqueur","fem": "moqueuse","masc_plur": "moqueurs","fem_plur": "moqueuses"},
    "vainqueur":{"masc": "vainqueur","fem": "vainqueuse","masc_plur": "vainqueurs","fem_plur": "vainqueuses"},
    "humeur":   {"masc": "humeur", "fem": "humeur",  "masc_plur": "humeurs", "fem_plur": "humeurs"},  # noun, skip
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Injection target size
INATTENTIVE_SAMPLE_SIZE = 50000
RANDOM_SEED = 42
MAX_SEQ_LEN = 64  # tokens after BPE

# BPE
BPE_VOCAB_SIZE = 8000