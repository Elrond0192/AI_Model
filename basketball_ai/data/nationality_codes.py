"""Canonicalize player nationality values to ISO 3166-1 alpha-2 codes.

The source DB contains both ISO alpha-2 and alpha-3 codes, plus common
basketball/IOC aliases (for example GER and DEU for Germany). Code matching is
exact after normalization; human-readable nationality labels remain supported.
"""
from __future__ import annotations

_COUNTRY_PAIRS = """
ABW AW AFG AF AGO AO AIA AI ALA AX ALB AL AND AD ARE AE ARG AR ARM AM ASM AS ATA AQ ATF TF ATG AG AUS AU AUT AT AZE AZ BDI BI BEL BE BEN BJ BES BQ BFA BF BGD BD BGR BG BHR BH BHS BS BIH BA BLM BL BLR BY BLZ BZ BMU BM BOL BO BRA BR BRB BB BRN BN BTN BT BVT BV BWA BW CAF CF CAN CA CCK CC CHE CH CHL CL CHN CN CIV CI CMR CM COD CD COG CG COK CK COL CO COM KM CPV CV CRI CR CUB CU CUW CW CXR CX CYM KY CYP CY CZE CZ DEU DE DJI DJ DMA DM DNK DK DOM DO DZA DZ ECU EC EGY EG ERI ER ESH EH ESP ES EST EE ETH ET FIN FI FJI FJ FLK FK FRA FR FRO FO FSM FM GAB GA GBR GB GEO GE GGY GG GHA GH GIB GI GIN GN GLP GP GMB GM GNB GW GNQ GQ GRC GR GRD GD GRL GL GTM GT GUF GF GUM GU GUY GY HKG HK HMD HM HND HN HRV HR HTI HT HUN HU IDN ID IMN IM IND IN IOT IO IRL IE IRN IR IRQ IQ ISL IS ISR IL ITA IT JAM JM JEY JE JOR JO JPN JP KAZ KZ KEN KE KGZ KG KHM KH KIR KI KNA KN KOR KR KWT KW LAO LA LBN LB LBR LR LBY LY LCA LC LIE LI LKA LK LSO LS LTU LT LUX LU LVA LV MAC MO MAF MF MAR MA MCO MC MDA MD MDG MG MDV MV MEX MX MHL MH MKD MK MLI ML MLT MT MMR MM MNE ME MNG MN MNP MP MOZ MZ MRT MR MSR MS MTQ MQ MUS MU MWI MW MYS MY MYT YT NAM NA NCL NC NER NE NFK NF NGA NG NIC NI NIU NU NLD NL NOR NO NPL NP NRU NR NZL NZ OMN OM PAK PK PAN PA PCN PN PER PE PHL PH PLW PW PNG PG POL PL PRI PR PRK KP PRT PT PRY PY PSE PS PYF PF QAT QA REU RE ROU RO RUS RU RWA RW SAU SA SDN SD SEN SN SGP SG SGS GS SHN SH SJM SJ SLB SB SLE SL SLV SV SMR SM SOM SO SPM PM SRB RS SSD SS STP ST SUR SR SVK SK SVN SI SWE SE SWZ SZ SXM SX SYC SC SYR SY TCA TC TCD TD TGO TG THA TH TJK TJ TKL TK TKM TM TLS TL TON TO TTO TT TUN TN TUR TR TUV TV TWN TW TZA TZ UGA UG UKR UA UMI UM URY UY USA US UZB UZ VAT VA VCT VC VEN VE VGB VG VIR VI VNM VN VUT VU WLF WF WSM WS YEM YE ZAF ZA ZMB ZM ZWE ZW ANG AO BUL BG CRO HR DEN DK ENG GB GER DE GRE GR IRI IR KOS XK NED NL PHI PH ROM RO RSA ZA SCO GB SLO SI SUI CH TPE TW WAL GB PUR PR
"""
_COUNTRY_TOKENS = _COUNTRY_PAIRS.split()
_ISO3_TO_ALPHA2 = dict(zip(_COUNTRY_TOKENS[::2], _COUNTRY_TOKENS[1::2]))

# Common basketball / IOC / legacy aliases not represented by ISO alpha-3.
_LEGACY_TO_ALPHA2 = {
    "ANG": "AO", "BUL": "BG", "CRO": "HR", "DEN": "DK", "ENG": "GB",
    "GER": "DE", "GRE": "GR", "IRI": "IR", "KOS": "XK", "NED": "NL",
    "PHI": "PH", "ROM": "RO", "RSA": "ZA", "SCO": "GB", "SLO": "SI",
    "SUI": "CH", "TPE": "TW", "WAL": "GB", "PUR": "PR",
}

_COUNTRY_NAMES_TO_ALPHA2 = {
    "GERMANY": "DE", "GERMAN": "DE", "DEUTSCHLAND": "DE",
    "ITALY": "IT", "ITALIAN": "IT", "ITALIA": "IT",
    "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US", "AMERICAN": "US",
    "SPAIN": "ES", "SPANISH": "ES", "ESPANA": "ES", "ESPAÑA": "ES",
    "FRANCE": "FR", "FRENCH": "FR", "GREECE": "GR", "GREEK": "GR",
    "SERBIA": "RS", "SERBIAN": "RS", "CROATIA": "HR", "CROATIAN": "HR",
    "SLOVENIA": "SI", "SLOVENIAN": "SI", "LITHUANIA": "LT", "LITHUANIAN": "LT",
    "LATVIA": "LV", "LATVIAN": "LV", "ESTONIA": "EE", "ESTONIAN": "EE",
    "TURKEY": "TR", "TURKISH": "TR", "ISRAEL": "IL", "ISRAELI": "IL",
    "SERBIA AND MONTENEGRO": "RS", "RUSSIA": "RU", "RUSSIAN": "RU",
    "POLAND": "PL", "POLISH": "PL", "PORTUGAL": "PT", "PORTUGUESE": "PT",
    "SWITZERLAND": "CH", "SWISS": "CH", "UNITED KINGDOM": "GB", "BRITISH": "GB",
    "CANADA": "CA", "CANADIAN": "CA", "AUSTRALIA": "AU", "AUSTRALIAN": "AU",
    "BRAZIL": "BR", "BRAZILIAN": "BR", "ARGENTINA": "AR", "ARGENTINIAN": "AR",
    "MEXICO": "MX", "MEXICAN": "MX", "ANGOLA": "AO", "ANGOLAN": "AO",
    "CAMEROON": "CM", "CAMEROONIAN": "CM", "CHINA": "CN", "CHINESE": "CN",
    "JAPAN": "JP", "JAPANESE": "JP", "NEW ZEALAND": "NZ", "NEW ZEALANDER": "NZ",
    "PUERTO RICO": "PR", "DOMINICAN REPUBLIC": "DO", "UKRAINE": "UA", "UKRAINIAN": "UA",
    "SWEDEN": "SE", "SWEDISH": "SE", "FINLAND": "FI", "FINNISH": "FI",
    "DENMARK": "DK", "DANISH": "DK", "BELGIUM": "BE", "BELGIAN": "BE",
}


def normalize_nationality(value: object) -> str:
    """Return a canonical alpha-2 code for known country codes/names."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in _ISO3_TO_ALPHA2:
        return _ISO3_TO_ALPHA2[upper]
    if upper in _LEGACY_TO_ALPHA2:
        return _LEGACY_TO_ALPHA2[upper]
    if upper in _COUNTRY_NAMES_TO_ALPHA2:
        return _COUNTRY_NAMES_TO_ALPHA2[upper]
    if len(upper) == 2 and upper.isascii() and upper.isalpha():
        return upper
    return raw


def is_nationality_code(value: object) -> bool:
    raw = str(value or "").strip().upper()
    return (
        raw in _ISO3_TO_ALPHA2
        or raw in _LEGACY_TO_ALPHA2
        or (len(raw) == 2 and raw.isascii() and raw.isalpha())
    )


def nationality_matches(requested: object, actual: object) -> bool:
    """Match codes exactly after normalization and country names semantically."""
    wanted = str(requested or "").strip()
    found = str(actual or "").strip()
    if not wanted or not found:
        return False
    wanted_norm = normalize_nationality(wanted)
    found_norm = normalize_nationality(found)
    if is_nationality_code(wanted):
        return wanted_norm.casefold() == found_norm.casefold()
    return wanted_norm.casefold() == found_norm.casefold() or wanted_norm.casefold() in found_norm.casefold()
