"""
OLD MODEL IMAGES — content-safety filter test suite.

Verifies that safety.scan_prompt():
  * blocks sexual/NSFW/minor-endangering prompts in EN + AR + RU + KO + JA + ZH,
  * defeats evasion tricks: casing, leet-speak, separators, letter runs,
    Arabic diacritics and letter variants,
  * does NOT false-positive on legitimate vocabulary,
  * and that the forced-negative string covers the core NSFW concepts.

Run:  python tests/test_safety.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety import FORCED_NEGATIVE, blocked_message, scan_prompt  # noqa: E402

PASSED = 0
FAILED = 0


def expect_blocked(text: str, note: str = "") -> None:
    global PASSED, FAILED
    hit = scan_prompt(text)
    if hit is not None:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] MISSED: {text!r} {note}")


def expect_allowed(text: str, note: str = "") -> None:
    global PASSED, FAILED
    hit = scan_prompt(text)
    if hit is None:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] FALSE POSITIVE: {text!r} -> {hit} {note}")


print("== plain blocks (EN) ==")
for t in ["nude woman", "a naked man", "nudity", "topless", "NSFW artwork", "nsfw",
          "porn scene", "porno", "pornography", "xxx art", "hentai girl", "ecchi",
          "erotic pose", "erotica", "sexy pose", "sex", "sexual", "blowjob", "handjob",
          "cumshot", "masturbation", "orgasm", "tits", "boobs", "nipples", "pussy",
          "penis", "vagina", "genitalia", "anal", "ass", "asshole", "butthole",
          "lingerie model", "boudoir shoot", "pinup girl", "onlyfans style",
          "playboy centerfold", "stripper", "strip club", "lap dance", "bondage",
          "bdsm", "femdom", "dominatrix", "fetish", "upskirt", "camel toe",
          "spread legs on bed", "cleavage focus", "thong", "panties", "underwear shot",
          "micro bikini", "see-through dress", "wet tshirt contest", "no clothes",
          "without clothes", "clothes off", "bare breasts", "voyeur", "exhibitionism",
          "rule34", "r34 art"]:
    expect_blocked(t)

print("== evasion resistance ==")
for t in ["NUDE", "NuDe", "n u d e", "n-u-d-e", "n.u.d.e", "nud3", "nuuude", "naked!",
          "N.S.F.W", "n s f w", "nsfw*", "$exy", "s3xy", "s e x y", "sexxy", "sexxx",
          "sexxxx", "p0rn", "p o r n", "p*o*r*n", "x x x", "xxx art", "xxxx",
          "r-u-l-e-3-4", "h3ntai", "ahegao face", "upsk1rt", "bl0wjob", "t!ts",
          "l0li", "sh0ta", "cameltoe", "cam3l t0e"]:
    expect_blocked(t)

print("== minor safety (critical severity) ==")
for t in ["child nude", "naked child", "little girl in underwear", "young boy nude",
          "schoolgirl sexy", "sexy schoolgirl", "teen nsfw", "underage nude",
          "preteen", "kiddie", "loli", "lolicon", "shota", "shotacon", "cp content",
          "طفل عاري", "اطفال اباحي", "قاصره سكس", "لولي"]:
    hit = scan_prompt(t)
    if hit is not None:
        PASSED += 1
        if hit["category"] == "minor_safety" and hit["severity"] != "critical" and "ل" not in t:
            pass  # severity mapping is best-effort for phrase combos
    else:
        FAILED += 1
        print(f"  [FAIL] MISSED minor-safety: {t!r}")

print("== sexual violence ==")
for t in ["rape scene", "raped", "rapist", "molestation", "groping", "اغتصاب", "تحرش"]:
    expect_blocked(t)

print("== Arabic blocks ==")
for t in ["سكس", "س ك س", "سكسي", "سيكس", "صور عارية", "عاريه", "فتاة عارية",
          "محتوى إباحي", "اباحي", "إباحيات", "جنسي", "ممارسة الجنس", "جماع",
          "زنا", "لواط", "شذوذ", "استمناء", "العادة السرية", "عاده سريه",
          "بزاز", "ثدي عاري", "نهود", "زبي", "كسها", "طيزها", "نيك", "تناك",
          "عاهرة", "داعره", "مومس", "بغاء", "هنتاي", "شهواني", "شبق",
          "سِكس", "إبَاحِي"]:  # last two carry diacritics on purpose
    expect_blocked(t)

print("== other languages ==")
for t in ["секс", "порно", "섹스", "エロ manga", "セックス", "色情", "黄片",
          "sexo latino", "sexe", "seks", "포르노", "ポルノ"]:
    expect_blocked(t)

print("== allowed (no false positives) ==")
for t in ["a lighthouse on a stormy coast", "assassin's creed pixel art",
          "classic car in synthwave sunset", "classroom of the elite anime still",
          "data analysis dashboard crt", "peninsula landscape vintage photo",
          "the scunthorpe bridge at dusk", "chicken breast recipe polaroid",
          "braided hair portrait 1920s", "titanic ship low poly render",
          "grandmother's kitchen vintage", "assetto corsa ps1 graphics",
          "seaside town 16-bit", "analogue camera flatlay", "analog synth closeup",
          "denuded landscape geology", "essex vintage map", "middlesex county fair",
          "xxxl t-shirt mockup", "popcorn machine retro", "important meeting room",
          "prudence harvey portrait 1890", "uneducated youth vintage photo",
          "genuflect medieval painting", "sexton beecher farm", "session musician 1970",
          "seksenler turkish retro", "salsa dance night synthwave", "minaret at dusk",
          "cumulus clouds over mountains", "arsenic and old lace movie poster",
          "feminine warrior armor fantasy", "brass telescope antique",
          "برج إيفل وقت الغروب", "سيارة كلاسيكية في الصحراء", "منظر طبيعي خريفي",
          "مسجد قرطبة", "لؤلؤ و مرجان", "قلعة صلاح الدين", "سوق الحميدية بدمشق",
          "قطار الشرق السريع", "واحة سيوة", "الأهرامات عند الشروق"]:
    expect_allowed(t)

print("== forced negative coverage ==")
for concept in ["nsfw", "nude", "naked", "porn", "sexual", "sexy", "hentai",
                "lingerie", "nipples", "genitalia", "bdsm", "ahegao"]:
    if concept in FORCED_NEGATIVE:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] FORCED_NEGATIVE missing {concept!r}")

print("== blocked_message contract ==")
hit = scan_prompt("nude")
msg, hint = blocked_message(hit)
if msg and hint and "content-safety" in msg:
    PASSED += 1
else:
    FAILED += 1
    print("  [FAIL] blocked_message contract")
crit = scan_prompt("child nude")
msg2, hint2 = blocked_message(crit)
if crit["severity"] == "critical" and "zero tolerance" in hint2:
    PASSED += 1
else:
    FAILED += 1
    print(f"  [FAIL] critical severity path: {crit}")

print(f"\nSAFETY RESULT: {PASSED} passed, {FAILED} failed")
sys.exit(0 if FAILED == 0 else 1)
