"""Normalize nationality aliases to the codes used by HoopMetrics data.

Nationality values are not uniformly ISO alpha-2: the database uses basketball
codes such as ITA, USA and ESP, while Germany is stored as DE (not DEU).
Do not globally convert three-letter nationality codes to two-letter codes.
Only explicitly configured aliases are rewritten.
"""
from __future__ import annotations

# Explicit source aliases: canonical DB code is on the right.
_CODE_ALIASES = {
    "DEU": "DE",
    "GER": "DE",
}

# Human-readable labels mapped to common basketball/nationality feed codes.
# Explicit codes such as ITA, USA and ESP are preserved as they are.
_COUNTRY_NAMES_TO_CODE = {
    "GERMANY": "DE", "GERMAN": "DE", "DEUTSCHLAND": "DE",
    "ITALY": "ITA", "ITALIAN": "ITA", "ITALIA": "ITA",
    "UNITED STATES": "USA", "UNITED STATES OF AMERICA": "USA", "AMERICAN": "USA",
    "SPAIN": "ESP", "SPANISH": "ESP", "ESPANA": "ESP", "ESPAÑA": "ESP",
    "FRANCE": "FRA", "FRENCH": "FRA",
    "GREECE": "GRE", "GREEK": "GRE",
    "SERBIA": "SRB", "SERBIAN": "SRB", "CROATIA": "CRO", "CROATIAN": "CRO",
    "SLOVENIA": "SLO", "SLOVENIAN": "SLO", "LITHUANIA": "LTU", "LITHUANIAN": "LTU",
    "LATVIA": "LAT", "LATVIAN": "LAT", "ESTONIA": "EST", "ESTONIAN": "EST",
    "TURKEY": "TUR", "TURKISH": "TUR", "ISRAEL": "ISR", "ISRAELI": "ISR",
    "RUSSIA": "RUS", "RUSSIAN": "RUS",
    "POLAND": "POL", "POLISH": "POL", "PORTUGAL": "POR", "PORTUGUESE": "POR",
    "SWITZERLAND": "SUI", "SWISS": "SUI", "UNITED KINGDOM": "GBR", "BRITISH": "GBR",
    "CANADA": "CAN", "CANADIAN": "CAN", "AUSTRALIA": "AUS", "AUSTRALIAN": "AUS",
    "BRAZIL": "BRA", "BRAZILIAN": "BRA", "ARGENTINA": "ARG", "ARGENTINIAN": "ARG",
    "MEXICO": "MEX", "MEXICAN": "MEX", "ANGOLA": "ANG", "ANGOLAN": "ANG",
    "CAMEROON": "CMR", "CAMEROONIAN": "CMR", "CHINA": "CHN", "CHINESE": "CHN",
    "JAPAN": "JPN", "JAPANESE": "JPN", "NEW ZEALAND": "NZL", "NEW ZEALANDER": "NZL",
    "PUERTO RICO": "PUR", "DOMINICAN REPUBLIC": "DOM", "UKRAINE": "UKR", "UKRAINIAN": "UKR",
    "SWEDEN": "SWE", "SWEDISH": "SWE", "FINLAND": "FIN", "FINNISH": "FIN",
    "DENMARK": "DEN", "DANISH": "DEN", "BELGIUM": "BEL", "BELGIAN": "BEL",
}


def normalize_nationality(value: object) -> str:
    """Apply configured source aliases and otherwise preserve supplied codes."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in _CODE_ALIASES:
        return _CODE_ALIASES[upper]
    if upper in _COUNTRY_NAMES_TO_CODE:
        return _COUNTRY_NAMES_TO_CODE[upper]
    if len(upper) in (2, 3) and upper.isascii() and upper.isalpha():
        # Keep feed conventions (e.g. ITA) instead of forcing ISO alpha-2.
        return upper
    return raw


def is_nationality_code(value: object) -> bool:
    """Treat compact two- and three-letter codes as exact-match identifiers."""
    raw = str(value or "").strip().upper()
    return len(raw) in (2, 3) and raw.isascii() and raw.isalpha()


def nationality_matches(requested: object, actual: object) -> bool:
    """Match compact codes exactly after explicit alias normalization."""
    wanted = str(requested or "").strip()
    found = str(actual or "").strip()
    if not wanted or not found:
        return False
    wanted_norm = normalize_nationality(wanted)
    found_norm = normalize_nationality(found)
    if is_nationality_code(wanted):
        return wanted_norm.casefold() == found_norm.casefold()
    return (
        wanted_norm.casefold() == found_norm.casefold()
        or wanted_norm.casefold() in found_norm.casefold()
    )
