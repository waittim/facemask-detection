#!/usr/bin/env python3
"""Generate WearMask docs i18n locale files from en.json master template."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "docs" / "i18n"

LANGUAGE_OPTIONS = {
    "en": "English",
    "es": "Español",
    "zh": "中文",
    "fr": "Français",
    "de": "Deutsch",
    "ja": "日本語",
    "pt": "Português",
    "ko": "한국어",
    "it": "Italiano",
    "ru": "Русский",
}

FEEDBACK_MAILTO = {
    "fr": "mailto:wearmask.feedback@gmail.com?subject=Commentaires&body=Si%20vous%20avez%20une%20capture%20d%27%C3%A9cran%20du%20r%C3%A9sultat%20incorrect,%20veuillez%20la%20joindre%20ici.",
    "de": "mailto:wearmask.feedback@gmail.com?subject=Feedback&body=Bitte%20f%C3%BCgen%20Sie%20hier%20einen%20Screenshot%20des%20fehlerhaften%20Ergebnisses%20bei.",
    "ja": "mailto:wearmask.feedback@gmail.com?subject=%E3%83%95%E3%82%A3%E3%83%BC%E3%83%89%E3%83%90%E3%83%83%E3%82%AF&body=%E8%AA%A4%E8%AA%8D%E8%A8%98%E9%A8%93%E3%81%AE%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%E3%81%8C%E3%81%82%E3%82%8C%E3%81%B0%E3%81%93%E3%81%A1%E3%82%89%E3%81%AB%E8%B2%BC%E3%82%8A%E4%BB%98%E3%81%91%E3%81%A6%E3%81%8F%E3%81%A0%E3%81%95%E3%81%84%E3%80%82",
    "pt": "mailto:wearmask.feedback@gmail.com?subject=Feedback&body=Se%20houver%20uma%20captura%20de%20tela%20do%20resultado%20incorreto,%20anexe-a%20aqui.",
    "ko": "mailto:wearmask.feedback@gmail.com?subject=%ED%94%BC%EB%93%9C%EB%B0%B1&body=%EC%98%A4%EC%8B%9D%EB%8F%99%20%EA%B2%B0%EA%B3%BC%EC%9D%98%20%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%EC%9D%B4%20%EC%9E%88%EB%8B%A4%EB%A9%B4%20%EC%97%AC%EA%B8%B0%EC%97%90%20%EB%B6%99%EC%97%AC%20%EB%84%A3%EC%96%B4%20%EC%A3%BC%EC%84%B8%EC%9A%94.",
    "it": "mailto:wearmask.feedback@gmail.com?subject=Feedback&body=Se%20disponi%20di%20uno%20screenshot%20del%20risultato%20errato,%20allegalo%20qui.",
    "ru": "mailto:wearmask.feedback@gmail.com?subject=%D0%9E%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D0%B0%D1%8F%20%D1%81%D0%B2%D1%8F%D0%B7%D1%8C&body=%D0%95%D1%81%D0%BB%D0%B8%20%D0%B5%D1%81%D1%82%D1%8C%20%D1%81%D0%BA%D1%80%D0%B8%D0%BD%D1%88%D0%BE%D1%82%20%D0%BD%D0%B5%D0%B2%D0%B5%D1%80%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%D0%B0,%20%D0%BF%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B8%D1%82%D0%B5%20%D0%B5%D0%B3%D0%BE%20%D0%B7%D0%B4%D0%B5%D1%81%D1%8C.",
}

PAPER_CITATION = (
    "Wang, Z., Wang, P., Louis, P. C., Wheless, L. E., & Huo, Y. (2023). "
    "WearMask: Fast in-browser face mask detection with serverless edge computing for COVID-19. "
    "<em>Electronic Imaging</em>, 35(11). https://doi.org/10.2352/EI.2023.35.11.HPCI-229"
)

PAPER_NAME = (
    "WearMask: Fast in-browser face mask detection with serverless edge computing for COVID-19"
)

TARGET_LOCALES = ["fr", "de", "ja", "pt", "ko", "it", "ru"]


def flatten_keys(obj: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.update(flatten_keys(value, path))
        else:
            keys.add(path)
    return keys


def deep_set(data: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = data
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def reorder_sections(data: dict) -> dict:
    """Ensure toast appears immediately after theme."""
    keys = list(data.keys())
    if "toast" in keys:
        keys.remove("toast")
    ordered: dict = {}
    for key in keys:
        ordered[key] = data[key]
        if key == "theme":
            ordered["toast"] = data["toast"]
    return ordered


def apply_preserved_values(data: dict, locale: str) -> None:
    data["hero"]["title"] = "WearMask"
    data["footer"]["name"] = "Zekun Wang"
    data["privacy"]["linkHref"] = "privacy/"
    data["language"]["options"] = LANGUAGE_OPTIONS.copy()
    data["citationPage"]["paperCitation"] = PAPER_CITATION
    data["researchPage"]["paperName"] = PAPER_NAME
    data["feedback"]["mailto"] = FEEDBACK_MAILTO[locale]

    labels = list(data["detection"]["labels"])
    labels[0] = "background"
    data["detection"]["labels"] = labels


def build_locale(base: dict, locale: str, overrides: dict) -> dict:
    data = deepcopy(base)
    for path, value in overrides.items():
        deep_set(data, path, value)
    apply_preserved_values(data, locale)
    return reorder_sections(data)


def validate_all() -> bool:
    en_keys = flatten_keys(json.loads((I18N_DIR / "en.json").read_text(encoding="utf-8")))
    ok = True
    print(f"\nReference key count (en): {len(en_keys)}")
    for loc in ["en", "zh", "es", *TARGET_LOCALES]:
        path = I18N_DIR / f"{loc}.json"
        if not path.exists():
            print(f"{loc}: MISSING FILE")
            ok = False
            continue
        keys = flatten_keys(json.loads(path.read_text(encoding="utf-8")))
        missing = en_keys - keys
        extra = keys - en_keys
        print(f"{loc} {len(keys)} missing {len(missing)} extra {len(extra)}")
        if missing:
            print(" ", sorted(missing)[:5])
            ok = False
        if extra:
            print("  extra sample:", sorted(extra)[:5])
            ok = False
    return ok


def load_bundles() -> dict[str, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_all_locale_bundles import bundle  # noqa: WPS433

    return {locale: bundle(locale) for locale in TARGET_LOCALES}


def main() -> int:
    bundles = load_bundles()

    with (I18N_DIR / "en.json").open(encoding="utf-8") as f:
        base = json.load(f)

    written: list[str] = []
    for locale in TARGET_LOCALES:
        overrides = bundles[locale]
        out = build_locale(base, locale, overrides)
        path = I18N_DIR / f"{locale}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write("\n")
        written.append(str(path.relative_to(ROOT)))
        print(f"Wrote {path.name}")

    if not validate_all():
        return 1
    print("\nAll locales match en.json key structure.")
    print(f"Key count: {len(flatten_keys(base))}")
    print("Files written:")
    for name in written:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
