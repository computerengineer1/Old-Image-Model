"""
OLD MODEL IMAGES — content safety module.

Layer 1 of the NSFW defense: a deterministic prompt filter with
bypass-resistant normalization. It is intentionally aggressive — this app is
a retro-art generator, there is no legitimate use for sexual content here.

Matching strategy (three passes, in order):
  1. PHRASE_PATTERNS  — multi-word regex combos (e.g. minor + sexual term).
  2. WORD_TERMS       — word-boundary match on a "folded" copy of the text
                        (lowercase, leet-fold, Arabic diacritics removed,
                         Arabic letter variants unified). Word boundaries keep
                        "assassin", "classic", "peninsula" safe.
  3. SUBSTRING_TERMS  — distinctive terms (>=5 chars or unambiguous) matched
                        inside a fully "squashed" copy of the text (all
                        non-letters removed, repeated letters collapsed) to
                        defeat "n s f w", "p*o$r)n", "sexxx" style evasions.

This module is pure Python (no torch/flask imports) so it can be unit-tested
anywhere and used from both app.py and generator.py.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i", "|": "l", "(": "c", "*": "",
})

_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640\u064F\u0650\u0651\u0652]")
_ARABIC_VARIANTS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",   # alef variants
    "ة": "ه",                                   # taa marbuta
    "ى": "ي",                                   # alef maqsura
    "ؤ": "و", "ئ": "ي",                         # hamza forms
    "گ": "ك", "پ": "ب", "چ": "ج",               # common non-MSA letters
})


def _fold(text: str) -> str:
    """Lowercase, strip accents/diacritics, leet-fold, unify Arabic variants."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().translate(_LEET_MAP).translate(_ARABIC_VARIANTS)
    text = _ARABIC_DIACRITICS.sub("", text)
    return text


def _squash(folded: str) -> str:
    """Remove every non-letter and collapse letter runs: 'n--u de' -> 'nude'."""
    letters_only = "".join(ch for ch in folded if ch.isalpha())
    return re.sub(r"(.)\1+", r"\1", letters_only)


# --------------------------------------------------------------------------- #
# Term lists (edit freely — data, not code)
# --------------------------------------------------------------------------- #

WORD_TERMS: dict[str, list[str]] = {
    "minor_safety": [
        # Highest severity: anything sexualizing minors. Also see PHRASE_PATTERNS.
        "loli", "lolicon", "shota", "shotacon", "preteen", "underage", "kiddie",
        "cp",
        "لولي", "لوليكون", "شوتا", "شوتاكون", "قاصره", "اطفال اباح",
    ],
    "sexual_violence": [
        "rape", "raped", "raping", "rapist", "noncon", "molest", "molestation",
        "groping", "groped", "fondling", "اغتصاب", "اغتصب", "تحرش", "اعتداء جنسي",
    ],
    "pornography": [
        "porn", "porno", "pornography", "pornographic", "pornstar", "xxx",
        "nsfw", "r34", "rule34", "hentai", "ecchi", "eromanga", "ero",
        "بورن", "بورنو", "اباحي", "اباحيه", "الاباحي", "اباحيات", "هنتاي", "ياوي",
    ],
    "nudity": [
        "nude", "nudes", "nudity", "naked", "unclothed", "topless", "bottomless",
        "undress", "undressed", "undressing", "pantsless", "starkers",
        "عاري", "عاريه", "عاريا", "العاري", "عرايا", "عري", "العري", "تعري",
        "تعريه", "التعري", "معرى", "معرين",
    ],
    "sexual_content": [
        "sex", "sexual", "sexually", "sexy", "sexiest", "erotic", "erotica",
        "lewd", "lascivious", "salacious", "seductive", "seductress",
        "intercourse", "orgasm", "masturbation", "masturbate", "masturbating",
        "wank", "wanking", "blowjob", "handjob", "footjob", "rimjob",
        "cunnilingus", "fellatio", "anilingus", "cumshot", "bukkake", "gokkun",
        "creampie", "facial cumshot", "semen", "ahegao", "futanari",
        "threesome", "orgy", "gangbang", "incest", "incestuous",
        "sodomy", "sodomize", "anal", "anally", "asshole", "butthole",
        "pussy", "dick", "penis", "vagina", "vulva", "clit", "clitoris",
        "testicle", "testicles", "scrotum", "genitalia", "genitals", "pubic",
        "boob", "boobs", "boobies", "tits", "titjob", "nipple", "nipples",
        "areola", "areolas", "breasts", "bare breast", "ass", "booty", "butt",
        "cum", "cumming", "cunt", "fanny", "kinky", "kink", "fetish",
        "bondage", "bdsm", "femdom", "dominatrix", "submissive slut",
        "voyeur", "voyeurism", "exhibitionism", "upskirt", "downblouse",
        "camel toe", "cameltoe", "cleavage", "lingerie", "underwear",
        "panties", "panty", "thong", "garter", "pasties", "bra", "bras",
        "boudoir", "pinup", "playboy", "centerfold", "stripper", "striptease",
        "stripclub", "lapdance", "peepshow", "brothel", "prostitute",
        "prostitution", "whore", "slut", "slutty", "hoe", "ممارسة الجنس",
        "سكس", "سكسي", "سكسيه", "السكس", "سيكس", "جنسي", "جنسيا", "جنسيه",
        "الجنسي", "جماع", "الجماع", "مضاجعه", "مضاجع", "زنا", "الزنا", "زاني",
        "لواط", "اللواط", "لوطي", "شذوذ", "الشذوذ", "استمناء", "الاستمناء",
        "عاده سريه", "شهوه", "شهواني", "شهوانيه", "شبق", "عاهره", "العاهره",
        "داعره", "الداعره", "مومس", "المومس", "بغاء", "ثدي", "ثديي", "الثدي",
        "اثداء", "نهد", "نهدي", "نهود", "النهود", "بزاز", "زب", "زبي", "الزب",
        "كس", "كسها", "كسي", "الكس", "طيز", "طيزها", "زموط", "نيك", "انيك",
        "تناك", "تنيك", "منكوح", "قضيبي", "خولي", "ايروتيك", "ماستربيشن",
    ],
    "sexual_content_i18n": [
        # other languages: Dutch/Turkish, Spanish, French, Russian, Korean,
        # Japanese, Chinese
        "seks", "sexo", "sexe", "секс", "섹스", "セックス", "性交", "色情",
        "黄片", "做爱", "エロ", "에로",
    ],
    "pornography_i18n": [
        "порно", "porno", "ポルノ", "포르노",
    ],
    "fetish_extreme": [
        "bestiality", "zoophilia", "necrophilia", "vore", "guro", "scat",
        "fisting", "gaping", "snuff", "watersports sex",
    ],
}

# Distinctive terms matched inside the squashed text (defeats separators/leet).
SUBSTRING_TERMS: dict[str, list[str]] = {
    "pornography": ["nsfw", "porn", "porno", "pornhub", "xnxx", "xvideos",
                     "onlyfans", "rule34", "hentai", "ecchi", "eromanga",
                     "youporn", "redtube"],
    "sexual_content": ["blowjob", "handjob", "footjob", "cumshot", "bukkake",
                        "ahegao", "futanari", "gangbang", "cunnilingus",
                        "fellatio", "anilingus", "masturbation", "camelt oe",
                        "cameltoe", "upskirt", "downblouse", "striptease",
                        "dominatrix", "voyeurism", "exhibitionism",
                        "spreadlegs", "sexscene", "makinglove", "no clothes",
                        "with out clothes", "clothesoff", "wet tshirt"],
    "nudity": ["naked", "nudes", "nude", "nudi", "topless", "bottomless",
                "عاريه", "التعري", "عرايا"],
    "sexual_content_2": ["sexy", "sexo", "sexe", "섹스", "セックス", "色情",
                          "stripclub", "lapdance", "peepshow", "camelt oe"],
    "sexual_violence": ["rape", "rapist"],
    "fetish_extreme": ["bestiality", "zoophilia", "necrophilia"],
    "sexual_content_ar": ["سكس", "سيكس", "اباحي", "اباحيه", "هنتاي", "استمناء",
                           "بزاز", "طيز", "زموط", "لواط", "شذوذ", "داعره",
                           "عاهره", "مومس", "نيكه", "تناك"],
    "minor_safety_ar": ["لولي", "لوليتا", "لوليكون", "شوتاكون"],
}
SUBSTRING_TERMS["sexual_content"] = [t.replace(" ", "") for t in SUBSTRING_TERMS["sexual_content"]]

# Proximity patterns: two concepts anywhere in one prompt = blocked.
PHRASE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sexual_content", re.compile(r"(?:ال)?عاده\s+(?:ال)?سريه|(?:ال)?ممارسه\s+(?:ال)?جنس")),
    ("sexual_content", re.compile(r"(?<!\w)sex{2,}(?!\w)|(?<!\w)nud{2,}(?!\w)|"
                                  r"(?<!\w)p+o+r+n+(?!\w)", re.IGNORECASE)),
    ("pornography", re.compile(r"(?<!\w)x{3,}(?!\w)|(?<!\w)(?:x[\s\-_.*]+){2,}x(?![\w])",
                               re.IGNORECASE)),
    ("minor_safety", re.compile(
        r"(child|kid|little\s+(girl|boy)|young\s+(girl|boy)|school\s*girl|school\s*boy|"
        r"preteen|teen(age|r)?|underage|minor|infant|toddler|loli|shota|"
        r"طفل|طفله|اطفال|قاصر|قاصره|صغير|صغيره|مراهق|مراهقه|بنات\s+صغيره)"
        r"[\s\S]{0,60}?"
        r"(nude|naked|nudity|sexy|seductive|lewd|erotic|erotica|nsfw|sexual|sex|"
        r"lingerie|underwear|panties|\bbra\b|bikini|suggestive|bed|undress|topless|"
        r"سكس|جنس|جنسي|عاري|تعري|اباح|شبق)", re.IGNORECASE)),
    ("minor_safety", re.compile(
        r"(nude|naked|sexy|seductive|lewd|erotic|nsfw|sexual|lingerie|panties|"
        r"سكس|جنسي|عاري|تعري|اباح)"
        r"[\s\S]{0,60}?"
        r"(child|kid|little\s+(girl|boy)|young\s+(girl|boy)|school\s*girl|"
        r"preteen|teen(age|r)?|underage|minor|loli|shota|"
        r"طفل|طفله|اطفال|قاصر|صغيره|مراهق)", re.IGNORECASE)),
    ("sexual_content", re.compile(
        r"(see\s*-?\s*through|wet\s+(t-?shirt|clothes|dress)|no\s+clothes|"
        r"without\s+clothes|clothes\s+off|taking\s+(it\s+)?off|"
        r"micro\s*bikini|string\s*bikini|breast\s+(feeding\s+)??nude|"
        r"bare\s+(breasts?|chest|ass|butt)|naked\s+\w+)", re.IGNORECASE)),
]

# Always merged into every negative prompt (Layer 2 of NSFW defense).
FORCED_NEGATIVE = (
    "nsfw, nude, nudity, naked, topless, explicit, sexual, sexy, seductive, "
    "porn, pornographic, xxx, hentai, ecchi, lewd, erotica, nipples, genitalia, "
    "lingerie, underwear, cleavage, bdsm, ahegao, spread legs, revealing clothes"
)


# --------------------------------------------------------------------------- #
# Compiled matchers
# --------------------------------------------------------------------------- #

def _word_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(_fold(term)) + r"(?!\w)")


_WORD_MATCHERS: list[tuple[str, str, re.Pattern]] = [
    (category, term, _word_pattern(term))
    for category, terms in WORD_TERMS.items()
    for term in terms
]

# Squashed-substring terms that must NOT match when preceded by these stems
# (e.g. "denuded" is a legitimate geology/forestry term).
_SUBSTR_LOOKBEHIND_EXCEPTIONS = {"nude": "(?<!de)", "nudi": "(?<!de)"}

_SUBSTR_MATCHERS: list[tuple[str, str, re.Pattern]] = [
    (category, term,
     re.compile(_SUBSTR_LOOKBEHIND_EXCEPTIONS.get(term, "") + re.escape(_squash(_fold(term)))))
    for category, terms in SUBSTRING_TERMS.items()
    for term in terms
]


def scan_prompt(text: str) -> Optional[dict]:
    """Return None if the text is acceptable, else
    {"category": str, "term": str, "severity": "critical"|"high"}.

    Severity 'critical' is reserved for minor-safety and sexual-violence hits.
    """
    if not text:
        return None
    folded = _fold(text)
    squashed = _squash(folded)

    for category, pattern in PHRASE_PATTERNS:
        # match both the raw and the folded text: Arabic patterns rely on the
        # folded form (taa-marbuta/hamza/diacritics unified), English ones on
        # the raw text.
        if pattern.search(text) or pattern.search(folded):
            return {"category": category, "term": "(phrase combination)",
                    "severity": "critical" if category == "minor_safety" else "high"}

    for category, term, matcher in _WORD_MATCHERS:
        if matcher.search(folded):
            return {"category": category, "term": term,
                    "severity": "critical" if category in ("minor_safety", "sexual_violence") else "high"}

    for category, term, squashed_matcher in _SUBSTR_MATCHERS:
        if squashed_matcher.search(squashed):
            sev_category = category
            for suffix in ("_ar", "_2", "_i18n"):
                if sev_category.endswith(suffix):
                    sev_category = sev_category[: -len(suffix)]
            return {"category": sev_category, "term": term,
                    "severity": "critical" if sev_category in ("minor_safety", "sexual_violence") else "high"}

    return None


def blocked_message(hit: dict) -> tuple[str, str]:
    """User-facing (message, hint) for a blocked prompt."""
    severity = hit.get("severity", "high")
    if severity == "critical":
        message = ("This prompt was permanently blocked by the content-safety filter "
                   f"(category: {hit['category']}).")
        hint = ("OLD MODEL IMAGES has zero tolerance for content that sexualizes minors "
                "or depicts sexual violence. This event was logged.")
    else:
        message = (f"This prompt was blocked by the content-safety filter "
                   f"(category: {hit['category']}).")
        hint = ("OLD MODEL IMAGES only creates retro/vintage artwork and refuses sexual or "
                "NSFW content in every language and spelling. Please rephrase your prompt.")
    return message, hint
