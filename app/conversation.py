import re
from typing import Any, Dict, List, Optional, Tuple

from .models import BeneficiaryProfile, ConversationAnalysis
from .ollama_client import call_ollama
from .knowledge_search import knowledge_searcher


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_FIELDS = [
    "name",
    "age",
    "education",
    "current_occupation",
    "skills",
    "interests",
    "aspirations",
    "employment_preference",
    "location",
    "language",
]


INVALID_VALUES = {
    "",
    "unknown",
    "not mentioned",
    "not provided",
    "none",
    "null",
    "n/a",
    "na",
}


# ============================================================
# LANGUAGE MARKERS
# ============================================================

HINGLISH_MARKERS = {
    "mera",
    "meri",
    "mere",
    "main",
    "mujhe",
    "mujhse",
    "hai",
    "hoon",
    "hun",
    "tha",
    "thi",
    "the",
    "ka",
    "ki",
    "ke",
    "karta",
    "karti",
    "karna",
    "chahta",
    "chahti",
    "pasand",
    "ruchi",
    "padhai",
    "padha",
    "seekha",
    "rehta",
    "rehti",
    "bolta",
    "bolti",
    "aata",
    "aati",
    "saal",
    "umar",
    "naam",
    "kaam",
    "apna",
    "apni",
    "future",
    "aage",
    "mein",
    "se",
    "par",
}


ENGLISH_MARKERS = {
    "my",
    "name",
    "i",
    "am",
    "is",
    "are",
    "years",
    "old",
    "studied",
    "study",
    "work",
    "working",
    "job",
    "skill",
    "skills",
    "interested",
    "interest",
    "want",
    "prefer",
    "live",
    "speak",
    "language",
    "business",
    "future",
    "education",
    "tailoring",
    "repair",
}


LANGUAGE_NAMES = {
    "hindi": "Hindi",
    "english": "English",
    "marathi": "Marathi",
    "bengali": "Bengali",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "gujarati": "Gujarati",
    "punjabi": "Punjabi",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "odia": "Odia",
    "urdu": "Urdu",
}


# ============================================================
# BASIC TEXT HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    return re.sub(
        r"\s+",
        " ",
        text.strip(),
    )


def normalize_for_match(text: str) -> str:
    text = normalize_text(text)

    return text.lower()


def clean_value(value: Any) -> Any:

    if value is None:
        return None

    if isinstance(value, str):

        value = normalize_text(value)

        if value.lower() in INVALID_VALUES:
            return None

        return value

    if isinstance(value, list):

        cleaned = []

        for item in value:

            item = clean_value(item)

            if item is not None:
                cleaned.append(item)

        return cleaned or None

    return value


def clean_extracted_phrase(value: str) -> str:

    if not value:
        return ""

    value = normalize_text(value)

    value = value.strip(
        " ,.;:!?।"
    )

    value = re.split(
        r"[.!?;।\n]",
        value,
    )[0]

    return value.strip(
        " ,.;:!?"
    )


def sentence_segments(text: str) -> List[str]:

    parts = re.split(
        r"[.!?;।,\n]+",
        normalize_text(text),
    )

    return [
        normalize_text(part)
        for part in parts
        if normalize_text(part)
    ]


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_input_language(text: str) -> str:

    text = normalize_text(text)

    if not text:
        return "unknown"

    devanagari_chars = len(
        re.findall(
            r"[\u0900-\u097F]",
            text,
        )
    )

    latin_chars = len(
        re.findall(
            r"[A-Za-z]",
            text,
        )
    )

    # Pure Devanagari.
    if devanagari_chars > 0 and latin_chars == 0:
        return "hindi"

    # Mostly Devanagari.
    if devanagari_chars > latin_chars:
        return "hindi"

    words = set(
        re.findall(
            r"[A-Za-z]+",
            text.lower(),
        )
    )

    hindi_score = len(
        words.intersection(
            HINGLISH_MARKERS
        )
    )

    english_score = len(
        words.intersection(
            ENGLISH_MARKERS
        )
    )

    if hindi_score >= 2 and english_score >= 1:
        return "hinglish"

    if hindi_score >= 2:
        return "hinglish"

    if english_score >= 2:
        return "english"

    return "unknown"


# ============================================================
# NAME
# ============================================================

def extract_name(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    patterns = [

        # Roman Hindi / Hinglish
        r"\bmera\s+naam\s+([A-Za-z][A-Za-z\s]{0,40}?)\s+hai\b",

        # English
        r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z\s]{0,40}?)(?:\s+hai)?\b",

        # Devanagari
        r"मेरा\s+नाम\s+([^\s,।.!?]+)\s+है",

        r"मेरा\s+नाम\s+([^\s,।.!?]+)\s+हैं",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            value = clean_extracted_phrase(
                match.group(1)
            )

            if value:

                return (
                    value,
                    1.0,
                    [match.group(0).strip()],
                )

    return None


# ============================================================
# AGE
# ============================================================

def extract_age(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    patterns = [

        # Hindi:
        # मैं 24 साल का हूं
        # मैं 24 साल का हूँ
        # मैं 24 साल की हूं
        # मैं 24 साल की हूँ
        r"मैं\s+(\d{1,3})\s*(?:साल|वर्ष)\s*(?:का|की)?\s*(?:हूं|हूँ)",

        # मेरी उम्र 24 साल है
        # मेरी उम्र 24 वर्ष है
        # मेरी उम्र 24 है
        r"मेरी\s+उम्र\s+(\d{1,3})\s*(?:साल|वर्ष)?\s*(?:है|हैं)",

        # उम्र 24 साल है
        r"उम्र\s+(\d{1,3})\s*(?:साल|वर्ष)?\s*(?:है|हैं)",

        # मेरी उम्र है 24 साल
        r"मेरी\s+उम्र\s+(?:है\s+)?(\d{1,3})\s*(?:साल|वर्ष)",

        # मैं 24 का हूं
        r"मैं\s+(\d{1,3})\s*(?:का|की)\s*(?:हूं|हूँ)",

        # Roman Hindi:
        # meri age 24 hai
        r"\bmeri\s+age\s+(\d{1,3})\s*(?:years?|yrs?)?\s*(?:hai|है)\b",

        # Roman Hindi:
        # main 24 saal ka hoon
        r"\bmain\s+(\d{1,3})\s*(?:saal|varsh)\s*(?:ka|ki)?\s*(?:hoon|hun)\b",

        # Roman Hindi:
        # meri umar 24 saal hai
        r"\bmeri\s+umar\s+(\d{1,3})\s*(?:saal|years?)?\s*(?:hai)\b",

        # English:
        # I am 24 years old
        r"\bI\s+am\s+(\d{1,3})\s*(?:years?|yrs?)?\s*old\b",

        # English:
        # I'm 24 years old
        r"\bI\s*['’]?m\s+(\d{1,3})\s*(?:years?|yrs?)?\s*old\b",

        # English:
        # My age is 24
        r"\bmy\s+age\s+is\s+(\d{1,3})\s*(?:years?|yrs?)?\b",

        # English:
        # I am 24 years
        r"\bI\s+am\s+(\d{1,3})\s*(?:years?|yrs?)\b",

        # English:
        # I am 24
        r"\bI\s+am\s+(\d{1,3})\b",

        # Standalone spoken Hindi:
        # 24 साल का हूं
        r"\b(\d{1,3})\s*(?:साल|वर्ष)\s*(?:का|की)?\s*(?:हूं|हूँ)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        try:
            age = int(match.group(1))
        except (ValueError, TypeError):
            continue

        # Reasonable human age validation.
        if not 1 <= age <= 120:
            continue

        return (
            age,
            1.0,
            [match.group(0).strip()],
        )

    return None


# ============================================================
# EDUCATION
# ============================================================

def extract_education(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    # ---------------------------------------------------------
    # Normalize common spacing variations
    # ---------------------------------------------------------

    text = text.strip()

    # ---------------------------------------------------------
    # Helper: convert numeric education level to ordinal
    # ---------------------------------------------------------

    def ordinal(number: int) -> str:

        if number == 1:
            return "1st"

        if number == 2:
            return "2nd"

        if number == 3:
            return "3rd"

        return f"{number}th"


    # ---------------------------------------------------------
    # 1. ENGLISH NUMERIC FORMS
    # ---------------------------------------------------------

    numeric_patterns = [

        r"\b(\d{1,2})\s*(?:st|nd|rd|th)\s+pass\b",

        r"\b(\d{1,2})\s*(?:st|nd|rd|th)\s+till\b",

        r"\bstudied\s+till\s+(\d{1,2})\s*(?:st|nd|rd|th)\b",

        r"\bstudied\s+up\s+to\s+(\d{1,2})\s*(?:st|nd|rd|th)\b",

        r"\bi\s+have\s+studied\s+till\s+(\d{1,2})\s*(?:st|nd|rd|th)\b",

        r"\bi\s+studied\s+till\s+(\d{1,2})\s*(?:st|nd|rd|th)\b",

        r"\bi\s+have\s+studied\s+up\s+to\s+(\d{1,2})\s*(?:st|nd|rd|th)\b",

        r"\bclass\s+(\d{1,2})\b",

        r"\b(\d{1,2})\s*(?:st|nd|rd|th)\s+class\b",
    ]


    for pattern in numeric_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        try:
            number = int(match.group(1))
        except (ValueError, TypeError):
            continue

        if not 1 <= number <= 20:
            continue

        return (
            ordinal(number),
            1.0,
            [match.group(0).strip()],
        )


    # ---------------------------------------------------------
    # 2. HINDI NUMERIC FORMS
    #
    # Examples:
    # 5वीं
    # 5वी
    # 5 वीं
    # 5 वी
    # 7वीं
    # 10वीं
    # 12वीं
    # ---------------------------------------------------------

    hindi_numeric_patterns = [

        r"\b(\d{1,2})\s*वीं\s*(?:कक्षा|तक|पास)?",

        r"\b(\d{1,2})\s*वी\s*(?:कक्षा|तक|पास)?",

        r"कक्षा\s*(\d{1,2})",

        r"क्लास\s*(\d{1,2})",

        r"(\d{1,2})\s*कक्षा",
    ]


    for pattern in hindi_numeric_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        try:
            number = int(match.group(1))
        except (ValueError, TypeError):
            continue

        if not 1 <= number <= 20:
            continue

        return (
            ordinal(number),
            1.0,
            [match.group(0).strip()],
        )


    # ---------------------------------------------------------
    # 3. HINDI WORD FORMS
    #
    # IMPORTANT:
    # Multiple spoken/spelling variations are accepted.
    # ---------------------------------------------------------

    hindi_education = {

        # 5th
        "5th": [
            "पाँचवीं",
            "पांचवीं",
            "पाँचवी",
            "पांचवी",
            "पाँचवि",
            "पांचवि",
            "पाँच",
            "पांच",
        ],

        # 7th
        "7th": [
            "सातवीं",
            "सातवी",
            "सातवि",
            "सात",
        ],

        # 10th
        "10th": [
            "दसवीं",
            "दसवी",
            "दसवि",
            "दस",
        ],

        # 12th
        "12th": [
            "बारहवीं",
            "बारहवी",
            "बारहवि",
            "बारहवीं",
            "बारहवी",
            "बारह",
        ],
    }


    # ---------------------------------------------------------
    # 4. Context words
    #
    # We prefer education-related context so that:
    #
    # "मेरी उम्र 12 साल है"
    #
    # does NOT become 12th education.
    # ---------------------------------------------------------

    education_context_patterns = [
        "कक्षा",
        "क्लास",
        "तक पढ़",
        "तक पढ",
        "तक पढ़",
        "पढ़ाई",
        "पढाई",
        "पढ़ाई",
        "पास",
        "स्कूल",
        "स्कूलिंग",
        "पढ़ा",
        "पढ़ी",
        "पढा",
        "पढी",
        "पढ़ा",
        "पढ़ी",
        "शिक्षा",
        "studied",
        "study",
        "class",
        "school",
        "pass",
        "education",
    ]


    # ---------------------------------------------------------
    # 5. Check Hindi word forms
    # ---------------------------------------------------------

    for result, variants in hindi_education.items():

        for variant in variants:

            # Escape the Hindi phrase so regex treats it literally
            escaped_variant = re.escape(variant)

            # -------------------------------------------------
            # First: explicit education context
            # -------------------------------------------------

            context_pattern = (
                rf"{escaped_variant}"
                rf"\s*(?:कक्षा|तक|पास)"
            )

            match = re.search(
                context_pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:

                return (
                    result,
                    1.0,
                    [match.group(0).strip()],
                )


            # -------------------------------------------------
            # Reverse order:
            #
            # कक्षा दस
            # क्लास बारह
            # कक्षा पांच
            # -------------------------------------------------

            reverse_context_pattern = (
                rf"(?:कक्षा|क्लास)\s*"
                rf"{escaped_variant}"
            )

            match = re.search(
                reverse_context_pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:

                return (
                    result,
                    1.0,
                    [match.group(0).strip()],
                )


    # ---------------------------------------------------------
    # 6. Natural Hindi sentences
    #
    # Examples:
    #
    # मैंने दसवीं तक पढ़ाई की है
    # मैं बारहवीं तक पढ़ा हूं
    # पांचवी तक पढ़ा हूं
    # सातवीं तक पढ़ी हूं
    # ---------------------------------------------------------

    natural_patterns = {

        "5th": [
            r"(?:मैंने|मैं)?\s*(?:पाँचवीं|पांचवीं|पाँचवी|पांचवी)\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
            r"(?:मैंने|मैं)?\s*(?:पाँच|पांच)\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
        ],

        "7th": [
            r"(?:मैंने|मैं)?\s*(?:सातवीं|सातवी)\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
            r"(?:मैंने|मैं)?\s*सात\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
        ],

        "10th": [
            r"(?:मैंने|मैं)?\s*(?:दसवीं|दसवी)\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
            r"(?:मैंने|मैं)?\s*दस\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
        ],

        "12th": [
            r"(?:मैंने|मैं)?\s*(?:बारहवीं|बारहवी)\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
            r"(?:मैंने|मैं)?\s*बारह\s*तक\s*(?:पढ़ाई|पढाई|पढ़ाई|पढ़|पढ|पढ़)",
        ],
    }


    for result, patterns_list in natural_patterns.items():

        for pattern in patterns_list:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:

                return (
                    result,
                    1.0,
                    [match.group(0).strip()],
                )


    # ---------------------------------------------------------
    # Nothing detected
    # ---------------------------------------------------------

    return None

# ============================================================
# EXPERIENCE
# ============================================================

def extract_experience(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    patterns = [

        # Roman Hindi / English
        r"\b(\d+)\s+saal\s+(?:ka\s+)?experience\b",

        r"\bexperience\s+(?:of\s+)?(\d+)\s+years?\b",

        r"\b(\d+)\s+years?\s+of\s+experience\b",

        # Devanagari
        r"(\d+)\s+साल\s+(?:का\s+)?अनुभव",

        r"(\d+)\s+वर्ष\s+(?:का\s+)?अनुभव",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            return (
                f"{match.group(1)} years",
                1.0,
                [match.group(0).strip()],
            )

    return None


# ============================================================
# OCCUPATION
# ============================================================

OCCUPATION_BAD_TERMS = {
    "baat",
    "language",
    "conversation",
    "prefer",
    "pasand",
    "interest",
    "interested",
    "speak",
    "bolta",
    "bolti",
    "bolte",
    "naam",
    "age",
    "umar",
    "saal",
    "padhai",
    "study",
    "studied",
    "rehta",
    "rehti",
    "live",
    "future",
    "gaya",
    "gayi",
    "tha",
    "thi",
}


def occupation_candidate_is_valid(
    value: str,
) -> bool:

    value = normalize_for_match(value)

    if not value:
        return False

    if len(value) > 80:
        return False

    words = set(value.split())

    if words.intersection(
        OCCUPATION_BAD_TERMS
    ):
        return False

    if value.startswith(
        (
            "hindi ",
            "english ",
            "marathi ",
        )
    ):
        return False

    if re.search(
        r"\b\d+\s+saal\b",
        value,
    ):
        return False

    return True


def extract_current_occupation(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    candidates = []

    patterns = [

        # Roman Hindi:
        (
            r"\bmain\s+([^,.;!?।\n]+?)\s+ka\s+kaam\s+"
            r"(?:karta|karti)\s+(?:hoon|hun)\b",
            5,
        ),

        (
            r"\bmain\s+([^,.;!?।\n]+?)\s+"
            r"(?:karta|karti)\s+(?:hoon|hun)\b",
            3,
        ),

        # English
        (
            r"\bi\s+work\s+as\s+([^,.;!?।\n]+)",
            4,
        ),

        (
            r"\bi\s+work\s+in\s+([^,.;!?।\n]+)",
            4,
        ),

        (
            r"\bi\s+am\s+working\s+as\s+([^,.;!?।\n]+)",
            4,
        ),

        (
            r"\bi\s+do\s+([^,.;!?।\n]+)",
            2,
        ),

        # Devanagari
        (
            r"मैं\s+([^,.;!?।\n]+?)\s+का\s+काम\s+"
            r"(?:करता|करती)\s+(?:हूँ|हूं)",
            5,
        ),

        (
            r"मैं\s+([^,.;!?।\n]+?)\s+"
            r"(?:करता|करती)\s+(?:हूँ|हूं)",
            3,
        ),

        (
            r"मैं\s+([^,.;!?।\n]+?)\s+(?:हूँ|हूं)",
            2,
        ),
    ]

    for pattern, score in patterns:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            value = clean_extracted_phrase(
                match.group(1)
            )

            if not occupation_candidate_is_valid(
                value
            ):
                continue

            candidates.append(
                (
                    score,
                    match.start(),
                    value,
                    match.group(0).strip(),
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    _, _, value, evidence = candidates[0]

    return (
        value,
        1.0,
        [evidence],
    )


# ============================================================
# SKILLS
# ============================================================

def extract_skills(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    patterns = [

        # Roman Hindi
        (
            r"\bmujhe\s+([^,.;!?।\n]+?)\s+"
            r"(?:aati|aata|aate)\s+(?:hai|hain)\b"
        ),

        # Explicit skill sentence
        (
            r"\bmy\s+skills?\s+(?:are|include)\s+"
            r"([^,.;!?।\n]+)"
        ),

        (
            r"\bi\s+have\s+skills?\s+in\s+"
            r"([^,.;!?।\n]+)"
        ),

        # English
        (
            r"\bi\s+can\s+([^,.;!?।\n]+)"
        ),

        # Devanagari
        (
            r"मुझे\s+([^,.;!?।\n]+?)\s+"
            r"(?:आती|आता|आते)\s+(?:है|हैं)"
        ),

        # Explicit कौशल
        (
            r"(?:मेरे\s+)?(?:कौशल|हुनर)\s+"
            r"(?:हैं|है)\s*[:\-]?\s*"
            r"([^,.;!?।\n]+)"
        ),
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = clean_extracted_phrase(
            match.group(1)
        )

        if not value:
            continue

        # Avoid capturing unrelated clauses.
        bad_terms = {
            "नाम",
            "उम्र",
            "पढ़ाई",
            "रहता",
            "रहती",
            "इंदौर",
            "हिंदी",
            "अंग्रेज़ी",
            "अंग्रेजी",
        }

        value_words = set(
            value.split()
        )

        if value_words.intersection(
            bad_terms
        ):
            continue

        return (
            value,
            1.0,
            [match.group(0).strip()],
        )

    return None


# ============================================================
# INTEREST NORMALIZATION
# ============================================================

INTEREST_NORMALIZATION = {

    # Tailoring
    "silai": "tailoring",
    "सिलाई": "tailoring",
    "silai ka kaam": "tailoring",
    "सिलाई का काम": "tailoring",
    "kapde silna": "tailoring",
    "कपड़े सिलना": "tailoring",
    "garment making": "tailoring",
    "garment work": "tailoring",
    "कपड़े बनाना": "tailoring",
    "सिलाई-बुनाई": "tailoring",
    "सिलाई बुनाई": "tailoring",

    # Bakery
    "baking": "bakery",
    "cake banana": "bakery",
    "cake making": "bakery",
    "bakery items banana": "bakery",
    "बेकिंग": "bakery",
    "केक बनाना": "bakery",

    # Pottery
    "mitti ke bartan": "pottery",
    "mitti ke bartan banana": "pottery",
    "clay work": "pottery",
    "मिट्टी के बर्तन": "pottery",
    "मिट्टी के बर्तन बनाना": "pottery",

    # Welding
    "welding": "welding",
    "metal fabrication": "welding",
    "वेल्डिंग": "welding",
    "धातु का काम": "welding",

    # Electrician
    "electrical work": "electrician",
    "electrical repair": "electrician",
    "wiring": "electrician",
    "electrician": "electrician",
    "इलेक्ट्रिकल काम": "electrician",
    "इलेक्ट्रिकल रिपेयर": "electrician",
    "वायरिंग": "electrician",

    # Mobile repair
    "mobile repairing": "mobile repair",
    "mobile repair": "mobile repair",
    "मोबाइल रिपेयरिंग": "mobile repair",
    "मोबाइल रिपेयर": "mobile repair",

    # Plumbing
    "plumbing": "plumbing",
    "pipe fitting": "plumbing",
    "प्लंबिंग": "plumbing",
    "पाइप फिटिंग": "plumbing",
}


def normalize_interest_value(
    value: str,
) -> str:

    value = clean_extracted_phrase(
        value
    )

    if not value:
        return ""

    normalized = normalize_for_match(
        value
    )

    if normalized in INTEREST_NORMALIZATION:

        return INTEREST_NORMALIZATION[
            normalized
        ]

    if value in INTEREST_NORMALIZATION:

        return INTEREST_NORMALIZATION[
            value
        ]

    if (
        "welding" in normalized
        and (
            "fabrication" in normalized
            or "metal" in normalized
        )
    ):
        return "welding"

    if (
        "सिलाई" in value
        or "बुनाई" in value
    ):
        return "tailoring"

    if (
        "बेकिंग" in value
        or "केक" in value
    ):
        return "bakery"

    if (
        "मिट्टी" in value
        and "बर्तन" in value
    ):
        return "pottery"

    return value


# ============================================================
# INTERESTS
# ============================================================

def extract_interests(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    original_text = normalize_text(
        text
    )

    patterns = [

        # Roman Hindi / Hinglish
        (
            r"\bmujhe\s+(.+?)\s+mein\s+interest\s+hai\b",
            "interest",
        ),

        (
            r"\bmujhe\s+(.+?)\s+mein\s+interested\s+hai\b",
            "interest",
        ),

        (
            r"\bmain\s+(.+?)\s+mein\s+interested\s+"
            r"(?:hoon|hun)\b",
            "interested",
        ),

        (
            r"\bmujhe\s+(.+?)\s+ke\s+kaam\s+mein\s+"
            r"interest\s+hai\b",
            "interest",
        ),

        (
            r"\bmeri\s+(.+?)\s+mein\s+ruchi\s+hai\b",
            "ruchi",
        ),

        (
            r"\bmujhe\s+(.+?)\s+mein\s+ruchi\s+hai\b",
            "ruchi",
        ),

        (
            r"\bmujhe\s+(.+?)\s+karne\s+mein\s+ruchi\s+hai\b",
            "ruchi",
        ),

        # Pasand
        (
            r"\bmujhe\s+(.+?)\s+pasand\s+hai\b",
            "pasand",
        ),

        (
            r"\bmujhe\s+(.+?)\s+ka\s+kaam\s+pasand\s+hai\b",
            "pasand",
        ),

        (
            r"\bmujhe\s+(.+?)\s+karna\s+pasand\s+hai\b",
            "pasand",
        ),

        (
            r"\bmain\s+(.+?)\s+karna\s+pasand\s+"
            r"(?:karta|karti)\s+(?:hoon|hun)\b",
            "pasand",
        ),

        # English
        (
            r"\bi\s+am\s+interested\s+in\s+([^.!?;।\n]+)",
            "interested",
        ),

        (
            r"\bi'm\s+interested\s+in\s+([^.!?;।\n]+)",
            "interested",
        ),

        (
            r"\bi\s+have\s+an?\s+interest\s+in\s+"
            r"([^.!?;।\n]+)",
            "interest",
        ),

        (
            r"\bmy\s+interest\s+is\s+([^.!?;।\n]+)",
            "interest",
        ),

        (
            r"\bi\s+like\s+([^.!?;।\n]+)",
            "like",
        ),

        (
            r"\bi\s+enjoy\s+([^.!?;।\n]+)",
            "enjoy",
        ),

        # Devanagari
        (
            r"मुझे\s+(.+?)\s+में\s+दिलचस्पी\s+है",
            "interest",
        ),

        (
            r"मुझे\s+(.+?)\s+में\s+रुचि\s+है",
            "ruchi",
        ),

        (
            r"मुझे\s+(.+?)\s+पसंद\s+है",
            "pasand",
        ),

        (
            r"मुझे\s+(.+?)\s+करना\s+पसंद\s+है",
            "pasand",
        ),

        (
            r"मेरी\s+रुचि\s+(.+?)\s+में\s+है",
            "ruchi",
        ),

        (
            r"मेरी\s+दिलचस्पी\s+(.+?)\s+में\s+है",
            "interest",
        ),
    ]

    candidates = []

    for pattern, expression_type in patterns:

        for match in re.finditer(
            pattern,
            original_text,
            flags=re.IGNORECASE,
        ):

            if not match.groups():
                continue

            raw_value = clean_extracted_phrase(
                match.group(1)
            )

            if not raw_value:
                continue

            value = re.sub(
                r"\b(kaam|work)\s*$",
                "",
                raw_value,
                flags=re.IGNORECASE,
            ).strip()

            if not value:
                continue

            bad_terms = {
                "naam",
                "age",
                "umar",
                "saal",
                "padhai",
                "study",
                "studied",
                "rehta",
                "rehti",
                "live",
                "future",
                "income",
                "location",
            }

            value_words = set(
                normalize_for_match(
                    value
                ).split()
            )

            if value_words.intersection(
                bad_terms
            ):
                continue

            score_map = {
                "interest": 5,
                "interested": 5,
                "ruchi": 5,
                "pasand": 4,
                "like": 4,
                "enjoy": 4,
            }

            score = score_map.get(
                expression_type,
                1,
            )

            canonical_value = normalize_interest_value(
                value
            )

            if not canonical_value:
                continue

            candidates.append(
                (
                    score,
                    match.start(),
                    canonical_value,
                    match.group(0).strip(),
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    _, _, value, evidence = candidates[0]

    return (
        value,
        1.0,
        [evidence],
    )


# ============================================================
# ASPIRATIONS
# ============================================================

def extract_aspirations(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    patterns = [

        # Roman Hindi / Hinglish
        (
            r"\bfuture\s+mein\s+([^,.;!?।\n]+?)\s+"
            r"chahta\s+(?:hoon|hun)\b"
        ),

        (
            r"\bfuture\s+mein\s+([^,.;!?।\n]+?)\s+"
            r"chahti\s+(?:hoon|hun)\b"
        ),

        (
            r"\baage\s+main\s+([^,.;!?।\n]+?)\s+"
            r"chahta\s+(?:hoon|hun)\b"
        ),

        (
            r"\baage\s+main\s+([^,.;!?।\n]+?)\s+"
            r"chahti\s+(?:hoon|hun)\b"
        ),

        (
            r"\bmain\s+([^,.;!?।\n]+?)\s+karna\s+"
            r"chahta\s+(?:hoon|hun)\b"
        ),

        (
            r"\bmain\s+([^,.;!?।\n]+?)\s+karna\s+"
            r"chahti\s+(?:hoon|hun)\b"
        ),

        # English
        (
            r"\bi\s+want\s+to\s+([^,.;!?।\n]+)"
        ),

        (
            r"\bin\s+the\s+future\s+i\s+want\s+to\s+"
            r"([^,.;!?।\n]+)"
        ),

        # Devanagari
        (
            r"मैं\s+भविष्य\s+में\s+([^,.;!?।\n]+?)\s+"
            r"करना\s+चाहता\s+(?:हूँ|हूं)"
        ),

        (
            r"मैं\s+भविष्य\s+में\s+([^,.;!?।\n]+?)\s+"
            r"करना\s+चाहती\s+(?:हूँ|हूं)"
        ),

        (
            r"मैं\s+आगे\s+([^,.;!?।\n]+?)\s+"
            r"करना\s+चाहता\s+(?:हूँ|हूं)"
        ),

        (
            r"मैं\s+आगे\s+([^,.;!?।\n]+?)\s+"
            r"करना\s+चाहती\s+(?:हूँ|हूं)"
        ),

        # Devanagari "future"
        (
            r"फ्यूचर\s+में\s+([^,.;!?।\n]+?)\s+"
            r"करना\s+चाहता\s+(?:हूँ|हूं)"
        ),

        (
            r"फ्यूचर\s+में\s+([^,.;!?।\n]+?)\s+"
            r"करना\s+चाहती\s+(?:हूँ|हूं)"
        ),

        # Natural spoken Hindi:
        # मैं अपना खुद का सिलाई का काम शुरू करना चाहता हूं
        (
            r"मैं\s+([^,.;!?।\n]+?)\s+शुरू\s+करना\s+"
            r"चाहता\s+(?:हूँ|हूं)"
        ),

        (
            r"मैं\s+([^,.;!?।\n]+?)\s+शुरू\s+करना\s+"
            r"चाहती\s+(?:हूँ|हूं)"
        ),
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = clean_extracted_phrase(
            match.group(1)
        )

        if not value:
            continue

        return (
            value,
            1.0,
            [match.group(0).strip()],
        )

    return None


# ============================================================
# EMPLOYMENT PREFERENCE
# ============================================================

def extract_employment_preference(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    normalized = normalize_for_match(
        text
    )

    # --------------------------------------------------------
    # Self-employment
    # --------------------------------------------------------

    self_patterns = [

        r"\bself[- ]employment\b",
        r"\bapna\s+(?:khud\s+ka\s+)?business\b",
        r"\bapni\s+(?:khud\s+ki\s+)?shop\b",
        r"\bown\s+business\b",
        r"\bown\s+shop\b",

        # Devanagari
        r"अपना\s+(?:खुद\s+का\s+)?व्यवसाय",
        r"अपना\s+(?:खुद\s+का\s+)?बिजनेस",
        r"अपना\s+(?:खुद\s+का\s+)?काम",
        r"खुद\s+का\s+काम",
        r"स्वरोजगार",
    ]

    for pattern in self_patterns:

        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        segments = sentence_segments(
            text
        )

        intent_words = [
            "chahta",
            "chahti",
            "want",
            "prefer",
            "pasand",
            "karna",
            "start",
            "shuru",
            "future",
            "अपना",
            "खुद",
            "स्वरोजगार",
        ]

        for segment in segments:

            segment_normalized = normalize_for_match(
                segment
            )

            if not re.search(
                pattern,
                segment_normalized,
                flags=re.IGNORECASE,
            ):
                continue

            if not any(
                word in segment_normalized
                for word in intent_words
            ):
                continue

            return (
                "self-employment",
                1.0,
                [segment.strip()],
            )

        return (
            "self-employment",
            1.0,
            [match.group(0).strip()],
        )

    # --------------------------------------------------------
    # Wage employment
    # --------------------------------------------------------

    job_patterns = [

        r"\bnaukri\b",
        r"\bjob\s+karna\b",
        r"\bjob\s+prefer\b",
        r"\bprefer\s+(?:a\s+)?job\b",
        r"\bwage\s+employment\b",

        r"नौकरी",
        r"वेतन\s+वाला\s+काम",
        r"वेतन\s+रोजगार",
    ]

    for pattern in job_patterns:

        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match:

            return (
                "wage-employment",
                1.0,
                [match.group(0).strip()],
            )

    return None


# ============================================================
# LOCATION
# ============================================================

def extract_location(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    patterns = [

        # Roman Hindi
        (
            r"\bmain\s+([^,.;!?।\n]+?)\s+mein\s+"
            r"(?:rehta|rahta)\s+(?:hoon|hun)\b"
        ),

        (
            r"\bmain\s+([^,.;!?।\n]+?)\s+mein\s+"
            r"rehti\s+(?:hoon|hun)\b"
        ),

        # English
        (
            r"\bi\s+live\s+in\s+([^,.;!?।\n]+)"
        ),

        (
            r"\bi\s+stay\s+in\s+([^,.;!?।\n]+)"
        ),

        (
            r"\bi\s+am\s+from\s+([^,.;!?।\n]+)"
        ),

        # Devanagari
        (
            r"मैं\s+([^,.;!?।\n]+?)\s+में\s+"
            r"(?:रहता|रहती)\s+(?:हूँ|हूं)"
        ),
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            value = clean_extracted_phrase(
                match.group(1)
            )

            if value:

                return (
                    value,
                    1.0,
                    [match.group(0).strip()],
                )

    return None


# ============================================================
# LANGUAGE
# ============================================================

def extract_language(
    text: str,
) -> Optional[Tuple[Any, float, List[str]]]:

    normalized = normalize_for_match(
        text
    )

    explicit_patterns = [

        # English / Roman Hindi
        r"\bi\s+speak\s+([^.!?;।\n]+)",
        r"\bmeri\s+language\s+([^.!?;।\n]+)",
        r"\bmy\s+language\s+is\s+([^.!?;।\n]+)",
        r"\blanguage\s+is\s+([^.!?;।\n]+)",

        r"\bmain\s+([^.!?;।\n]+?)\s+"
        r"bolta\s+(?:hoon|hun)",

        r"\bmain\s+([^.!?;।\n]+?)\s+"
        r"bolti\s+(?:hoon|hun)",

        r"\bmain\s+([^.!?;।\n]+?)\s+mein\s+"
        r"baat\s+(?:karta|karti)\s+(?:hoon|hun)",

        # Devanagari
        r"मैं\s+([^.!?;।\n]+?)\s+"
        r"बोलता\s+(?:हूँ|हूं)",

        r"मैं\s+([^.!?;।\n]+?)\s+"
        r"बोलती\s+(?:हूँ|हूं)",

        r"मैं\s+([^.!?;।\n]+?)\s+में\s+"
        r"बात\s+(?:करता|करती)\s+(?:हूँ|हूं)",
    ]

    devanagari_language_names = {
        "हिंदी": "Hindi",
        "अंग्रेज़ी": "English",
        "अंग्रेजी": "English",
        "मराठी": "Marathi",
        "बंगाली": "Bengali",
        "तमिल": "Tamil",
        "तेलुगु": "Telugu",
        "गुजराती": "Gujarati",
        "पंजाबी": "Punjabi",
        "कन्नड़": "Kannada",
        "मलयालम": "Malayalam",
        "ओड़िया": "Odia",
        "उड़िया": "Odia",
        "उर्दू": "Urdu",
    }

    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        context = match.group(0)

        found_languages = []

        # Roman language names.
        for key, display_name in LANGUAGE_NAMES.items():

            if re.search(
                rf"\b{re.escape(key)}\b",
                context,
                flags=re.IGNORECASE,
            ):

                if display_name not in found_languages:

                    found_languages.append(
                        display_name
                    )

        # Devanagari language names.
        for key, display_name in (
            devanagari_language_names.items()
        ):

            if key in context:

                if display_name not in found_languages:

                    found_languages.append(
                        display_name
                    )

        if found_languages:

            return (
                found_languages,
                1.0,
                [context.strip()],
            )

    return None


# ============================================================
# DETERMINISTIC EXTRACTOR
# ============================================================

def deterministic_extract(
    text: str,
) -> Dict[str, Dict[str, Any]]:

    extracted: Dict[
        str,
        Dict[str, Any],
    ] = {}

    extractors = {

        "name": extract_name,

        "age": extract_age,

        "education": extract_education,

        "experience": extract_experience,

        "current_occupation": extract_current_occupation,

        "skills": extract_skills,

        "interests": extract_interests,

        "aspirations": extract_aspirations,

        "employment_preference": (
            extract_employment_preference
        ),

        "location": extract_location,

        "language": extract_language,
    }

    for field_name, extractor in (
        extractors.items()
    ):

        try:

            result = extractor(
                text
            )

        except Exception:
            # One extractor should never crash
            # the complete conversation engine.
            continue

        if result is None:
            continue

        value, confidence, evidence = result

        extracted[field_name] = {
            "value": value,
            "confidence": confidence,
            "evidence": evidence,
        }

    return extracted


# ============================================================
# LLM VALIDATION
# ============================================================

FIELD_CONTEXT_PATTERNS = {

    "name": [
        r"\bmera\s+naam\b",
        r"\bmy\s+name\b",
        r"\bname\s+is\b",
        r"मेरा\s+नाम",
    ],

    "age": [
        r"\bsaal\s+ka\s+(?:hoon|hun)\b",
        r"\bsaal\s+ki\s+(?:hoon|hun)\b",
        r"\byears?\s+old\b",
        r"\bage\b",
        r"\bumar\b",

        r"साल\s+का\s+हूँ",
        r"साल\s+की\s+हूँ",
        r"साल\s+का\s+हूं",
        r"साल\s+की\s+हूं",
        r"उम्र",
    ],

    "education": [
        r"\bpadhai\b",
        r"\bstudied\b",
        r"\bstudy\b",
        r"\bclass\b",
        r"\bgrade\b",
        r"\bpass\b",
        r"\bवीं\b",

        r"पढ़ाई",
        r"कक्षा",
        r"तक",
        r"पास",
    ],

    "current_occupation": [
        r"\bka\s+kaam\s+(?:karta|karti)\b",
        r"\b(?:karta|karti)\s+(?:hoon|hun)\b",
        r"\bwork\s+as\b",
        r"\bwork\s+in\b",
        r"\bworking\s+as\b",
        r"\bi\s+do\b",
        r"\boccupation\b",

        r"का\s+काम\s+(?:करता|करती)",
        r"(?:करता|करती)\s+हूँ",
        r"(?:करता|करती)\s+हूं",
        r"काम\s+करता",
        r"काम\s+करती",
    ],

    "skills": [
        r"\bmujhe\b.*\b(?:aati|aata|aate)\b",
        r"\bskills?\b",
        r"\bi\s+can\b",
        r"\bi\s+have\s+skills\b",

        r"मुझे\b.*(?:आती|आता|आते)",
        r"कौशल",
        r"हुनर",
    ],

    "interests": [
        r"\bmujhe\b.*\binterest\b",
        r"\bmujhe\b.*\binterested\b",
        r"\bmujhe\b.*\bpasand\b",
        r"\bmujhe\b.*\bruchi\b",
        r"\bmeri\b.*\bruchi\b",
        r"\bmain\b.*\binterested\b",
        r"\bmain\b.*\binterest\b",
        r"\bmain\b.*\bpasand\b",

        r"\bi\s+am\s+interested\s+in\b",
        r"\bi'm\s+interested\s+in\b",
        r"\bi\s+have\s+an?\s+interest\s+in\b",
        r"\bmy\s+interest\s+is\b",
        r"\bi\s+like\b",
        r"\bi\s+enjoy\b",

        r"रुचि",
        r"पसंद",
        r"दिलचस्पी",
    ],

    "aspirations": [
        r"\bfuture\b",
        r"\baage\b",
        r"\bwant\b",
        r"\bchahta\b",
        r"\bchahti\b",
        r"\bkarna\s+chahta\b",
        r"\bkarna\s+chahti\b",

        r"भविष्य",
        r"आगे",
        r"चाहता",
        r"चाहती",
    ],

    "employment_preference": [
        r"\bprefer\b",
        r"\bself[- ]employment\b",
        r"\bown\s+business\b",
        r"\bapna\s+business\b",
        r"\bnaukri\b",
        r"\bwage\s+employment\b",

        r"अपना\s+काम",
        r"अपना\s+व्यवसाय",
        r"अपना\s+बिजनेस",
        r"खुद\s+का\s+काम",
        r"स्वरोजगार",
        r"नौकरी",
    ],

    "location": [
        r"\bmein\s+rehta\b",
        r"\bmein\s+rehti\b",
        r"\blive\s+in\b",
        r"\bstay\s+in\b",
        r"\bfrom\b",

        r"में\s+रहता",
        r"में\s+रहती",
    ],

    "language": [
        r"\bspeak\b",
        r"\blanguage\b",
        r"\bbolta\b",
        r"\bbolti\b",
        r"\bmein\s+baat\b",
        r"\bbaat\s+karna\s+prefer\b",

        r"बोलता",
        r"बोलती",
        r"भाषा",
        r"बात\s+करता",
        r"बात\s+करती",
    ],

    "experience": [
        r"\bexperience\b",
        r"\bsaal\s+ka\s+experience\b",
        r"\byears?\s+of\s+experience\b",

        r"अनुभव",
    ],

    "traditional_occupation": [
        r"\bfamily\s+occupation\b",
        r"\bfamily\s+business\b",
        r"\btraditional\b",
        r"\bparivaar\b",
        r"\bkhandani\b",

        r"पारिवारिक",
        r"परिवार",
        r"पारंपरिक",
        r"खानदानी",
    ],

    "income_goal": [
        r"\bincome\b",
        r"\bkamai\b",
        r"\bearn\b",
        r"\bearning\b",
        r"\bincome\s+goal\b",

        r"कमाई",
        r"आय",
    ],

    "mobility": [
        r"\bmobility\b",
        r"\btravel\b",
        r"\bmove\b",
        r"\bdoor\b",
        r"\bpaas\b",

        r"आना",
        r"जाना",
        r"यात्रा",
    ],

    "work_constraints": [
        r"\bconstraint\b",
        r"\bproblem\b",
        r"\bcan't\b",
        r"\bcannot\b",
        r"\bdikkat\b",
        r"\bpareshani\b",

        r"दिक्कत",
        r"परेशानी",
        r"समस्या",
    ],

    "previous_training": [
        r"\btraining\b",
        r"\btrained\b",
        r"\bcourse\b",
        r"\bcertificate\b",

        r"प्रशिक्षण",
        r"ट्रेनिंग",
        r"कोर्स",
        r"प्रमाणपत्र",
    ],
}


def evidence_supported(
    evidence: Any,
    user_message: str,
) -> bool:

    if not isinstance(
        evidence,
        list,
    ):
        return False

    if not evidence:
        return False

    normalized_message = normalize_for_match(
        user_message
    )

    for item in evidence:

        if not isinstance(
            item,
            str,
        ):
            return False

        item = normalize_for_match(
            item
        )

        if not item:
            return False

        if item not in normalized_message:
            return False

    return True


def value_supported_by_evidence(
    field_name: str,
    value: Any,
    evidence: List[str],
) -> bool:

    if value is None:
        return False

    evidence_text = normalize_for_match(
        " ".join(evidence)
    )

    if isinstance(value, list):

        if not value:
            return False

        for item in value:

            item_text = normalize_for_match(
                str(item)
            )

            if field_name == "language":

                if item_text not in evidence_text:
                    return False

            elif item_text not in evidence_text:

                # Interest values may be canonicalized.
                if field_name == "interests":

                    normalized_interest = (
                        normalize_interest_value(
                            str(item)
                        )
                    )

                    if normalized_interest == "tailoring":

                        tailoring_terms = [
                            "tailoring",
                            "silai",
                            "सिलाई",
                            "बुनाई",
                            "garment",
                            "कपड़े",
                        ]

                        if any(
                            term in evidence_text
                            for term in tailoring_terms
                        ):
                            continue

                return False

        return True

    value_text = normalize_for_match(
        str(value)
    )

    # Employment preference.
    if field_name == "employment_preference":

        if value_text == "self-employment":

            return any(
                phrase in evidence_text
                for phrase in [
                    "self employment",
                    "self-employment",
                    "apna business",
                    "own business",
                    "apna shop",
                    "own shop",
                    "apna kaam",
                    "khud ka kaam",
                    "अपना काम",
                    "खुद का काम",
                    "स्वरोजगार",
                ]
            )

        if value_text == "wage-employment":

            return any(
                phrase in evidence_text
                for phrase in [
                    "job",
                    "naukri",
                    "wage employment",
                    "नौकरी",
                ]
            )

    # Interest normalization.
    if field_name == "interests":

        normalized_value = normalize_interest_value(
            str(value)
        )

        if normalized_value == "tailoring":

            tailoring_terms = [
                "tailoring",
                "silai",
                "garment",
                "कपड़े",
                "सिलाई",
                "बुनाई",
            ]

            return any(
                term in evidence_text
                for term in tailoring_terms
            )

        return (
            value_text in evidence_text
        )

    return (
        value_text in evidence_text
    )


def llm_context_supported(
    field_name: str,
    value: Any,
    user_message: str,
) -> bool:

    patterns = FIELD_CONTEXT_PATTERNS.get(
        field_name,
        [],
    )

    if not patterns:
        return False

    normalized_message = normalize_for_match(
        user_message
    )

    context_found = any(
        re.search(
            pattern,
            normalized_message,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )

    if not context_found:
        return False

    # Language must contain explicit language names.
    if field_name == "language":

        language_values = (
            value
            if isinstance(value, list)
            else [value]
        )

        language_name_map = {
            "hindi": [
                "hindi",
                "हिंदी",
            ],
            "english": [
                "english",
                "अंग्रेजी",
                "अंग्रेज़ी",
            ],
            "marathi": [
                "marathi",
                "मराठी",
            ],
            "bengali": [
                "bengali",
                "बंगाली",
            ],
            "tamil": [
                "tamil",
                "तमिल",
            ],
            "telugu": [
                "telugu",
                "तेलुगु",
            ],
            "gujarati": [
                "gujarati",
                "गुजराती",
            ],
            "punjabi": [
                "punjabi",
                "पंजाबी",
            ],
        }

        for language in language_values:

            language_name = normalize_for_match(
                str(language)
            )

            possible_names = language_name_map.get(
                language_name,
                [language_name],
            )

            if not any(
                name in normalized_message
                for name in possible_names
            ):
                return False

        return True

    return True


def validate_llm_field(
    field_name: str,
    field_data: Any,
    user_message: str,
) -> bool:

    if not isinstance(
        field_data,
        dict,
    ):
        return False

    value = clean_value(
        field_data.get("value")
    )

    confidence = field_data.get(
        "confidence",
        0,
    )

    evidence = field_data.get(
        "evidence",
        [],
    )

    try:

        confidence = float(
            confidence
        )

    except (
        TypeError,
        ValueError,
    ):

        return False

    if confidence < 0.80:
        return False

    if value is None:
        return False

    if not evidence_supported(
        evidence,
        user_message,
    ):
        return False

    if not value_supported_by_evidence(
        field_name,
        value,
        evidence,
    ):
        return False

    if not llm_context_supported(
        field_name,
        value,
        user_message,
    ):
        return False

    return True


# ============================================================
# CONVERSATION ENGINE
# ============================================================

class ConversationEngine:

    def __init__(self):

        self.profile = BeneficiaryProfile()

        self.history: List[
            Dict[str, str]
        ] = []

        self.response_language = "hindi"

        self.last_knowledge_results: Dict[
            str,
            Any,
        ] = {}

        self.last_llm_error: Optional[
            str
        ] = None

    # ========================================================
    # LANGUAGE
    # ========================================================

    def update_response_language(
        self,
        user_message: str,
    ):

        detected = detect_input_language(
            user_message
        )

        if detected != "unknown":

            self.response_language = (
                detected
            )

    # ========================================================
    # PROFILE
    # ========================================================

    def profile_dict(
        self,
    ) -> Dict[str, Any]:

        return self.profile.model_dump()

    def get_missing_fields(
        self,
    ) -> List[str]:

        missing = []

        for field_name in REQUIRED_FIELDS:

            field = getattr(
                self.profile,
                field_name,
            )

            if field.value is None:

                missing.append(
                    field_name
                )

        return missing

    def update_profile(
        self,
        extracted: Dict[
            str,
            Dict[str, Any],
        ],
    ) -> Dict[
        str,
        Dict[str, Any],
    ]:

        accepted = {}

        for field_name, field_data in (
            extracted.items()
        ):

            if (
                field_name
                not in self.profile.model_fields
            ):
                continue

            if not isinstance(
                field_data,
                dict,
            ):
                continue

            value = clean_value(
                field_data.get(
                    "value"
                )
            )

            if value is None:
                continue

            confidence = field_data.get(
                "confidence",
                0,
            )

            evidence = field_data.get(
                "evidence",
                [],
            )

            field = getattr(
                self.profile,
                field_name,
            )

            # Deterministic extraction is authoritative.
            field.value = value
            field.confidence = confidence
            field.evidence = evidence

            accepted[field_name] = {
                "value": value,
                "confidence": confidence,
                "evidence": evidence,
            }

        return accepted

    # ========================================================
    # LLM EXTRACTION
    # ========================================================

    def extract_with_llm(
        self,
        user_message: str,
    ) -> Dict[
        str,
        Dict[str, Any],
    ]:

        self.last_llm_error = None

        system_prompt = """
You are an information extraction engine for a
PM-AJAY livelihood assistant.

Extract information ONLY from the CURRENT USER MESSAGE.

Do NOT infer information.

Do NOT use common sense to fill missing fields.

Do NOT assume information that is not explicitly stated.

Every extracted field MUST have:
- value
- confidence
- evidence

Evidence MUST be an exact short quote from the CURRENT USER MESSAGE.

If a field is not explicitly stated, return null.

Return JSON only.

Fields:
name
age
education
current_occupation
traditional_occupation
experience
skills
interests
aspirations
income_goal
employment_preference
mobility
work_constraints
location
language
previous_training
"""

        user_prompt = f"""
CURRENT USER MESSAGE:

{user_message}

Return only the JSON extraction.
"""

        try:

            result = call_ollama(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
            )

            if not isinstance(
                result,
                dict,
            ):

                return {}

            return result

        except Exception as exc:

            self.last_llm_error = str(
                exc
            )

            return {}

    # ========================================================
    # NEXT QUESTION
    # ========================================================

    QUESTION_MAP = {

        # ====================================================
        # PURE HINDI
        # ====================================================

        "hindi": {

            "name":
                "आपका नाम क्या है?",

            "age":
                "आपकी उम्र कितनी है?",

            "education":
                "आपने कितनी पढ़ाई की है?",

            "current_occupation":
                "आप अभी क्या काम करते हैं?",

            "skills":
                "आपको कौन-कौन से कौशल या काम आते हैं?",

            "interests":
                "आपको किस तरह के काम में रुचि है?",

            "aspirations":
                "आप भविष्य में क्या करना चाहते हैं?",

            "employment_preference":
                "आप नौकरी करना चाहते हैं या अपना व्यवसाय?",

            "location":
                "आप किस शहर या गाँव में रहते हैं?",

            "language":
                "आप किस भाषा में बात करना पसंद करते हैं?",
        },

        # ====================================================
        # HINGLISH
        # ====================================================

        "hinglish": {

            "name":
                "Aapka naam kya hai?",

            "age":
                "Aapki age kitni hai?",

            "education":
                "Aapne kitni padhai ki hai?",

            "current_occupation":
                "Aap abhi kya kaam karte hain?",

            "skills":
                "Aapko kaun-kaun se skills aati hain?",

            "interests":
                "Aapko kis kaam mein interest hai?",

            "aspirations":
                "Future mein aap kya karna chahte hain?",

            "employment_preference":
                "Aap job karna chahte hain ya apna business?",

            "location":
                "Aap kis city ya village mein rehte hain?",

            "language":
                "Aap kis language mein baat karna prefer karte hain?",
        },

        # ====================================================
        # ENGLISH
        # ====================================================

        "english": {

            "name":
                "What is your name?",

            "age":
                "How old are you?",

            "education":
                "What is your highest level of education?",

            "current_occupation":
                "What work do you currently do?",

            "skills":
                "What skills do you have?",

            "interests":
                "What type of work are you interested in?",

            "aspirations":
                "What do you want to do in the future?",

            "employment_preference":
                "Do you prefer a job or self-employment?",

            "location":
                "Where do you currently live?",

            "language":
                "Which language do you prefer to speak?",
        },
    }

    def next_question(
        self,
    ) -> str:

        missing = self.get_missing_fields()

        if not missing:
            return ""

        language = self.response_language

        if language not in self.QUESTION_MAP:

            language = "hindi"

        field = missing[0]

        return self.QUESTION_MAP[
            language
        ].get(
            field,
            self.QUESTION_MAP[
                "hindi"
            ][field],
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    def build_response(
        self,
        missing_fields: List[str],
    ) -> str:

        # ----------------------------------------------------
        # Some required information is missing.
        # ----------------------------------------------------

        if missing_fields:

            return self.next_question()

        # ----------------------------------------------------
        # All required information is available.
        # ----------------------------------------------------

        if self.response_language == "english":

            return (
                "Thank you. Your information is complete. "
                "I will now find suitable training and "
                "livelihood options for you."
            )

        if self.response_language == "hinglish":

            return (
                "Dhanyavaad. Aapki information complete ho gayi hai. "
                "Ab main aapke liye suitable training aur "
                "livelihood options find karta hoon."
            )

        # Pure Hindi.

        return (
            "धन्यवाद। आपकी जानकारी पूरी हो गई है। "
            "अब मैं आपके लिए उपयुक्त प्रशिक्षण और "
            "आजीविका के विकल्प खोजूँगा।"
        )

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process_message(
        self,
        user_message: str,
    ) -> ConversationAnalysis:

        user_message = normalize_text(
            user_message
        )

        if not user_message:

            raise ValueError(
                "User message cannot be empty."
            )

        # ----------------------------------------------------
        # 1. Detect response language.
        # ----------------------------------------------------

        self.update_response_language(
            user_message
        )

        self.history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # ----------------------------------------------------
        # 2. Deterministic extraction FIRST.
        # ----------------------------------------------------

        deterministic_fields = (
            deterministic_extract(
                user_message
            )
        )

        accepted_fields = (
            self.update_profile(
                deterministic_fields
            )
        )

        # ----------------------------------------------------
        # 3. LLM extraction SECOND.
        #
        # LLM is fallback only.
        # Deterministic extraction wins.
        # ----------------------------------------------------

        llm_fields = (
            self.extract_with_llm(
                user_message
            )
        )

        if isinstance(
            llm_fields,
            dict,
        ):

            for field_name, field_data in (
                llm_fields.items()
            ):

                if (
                    field_name
                    not in self.profile.model_fields
                ):
                    continue

                # Deterministic extractor already
                # found this field.
                if (
                    field_name
                    in accepted_fields
                ):
                    continue

                if not validate_llm_field(
                    field_name,
                    field_data,
                    user_message,
                ):
                    continue

                value = clean_value(
                    field_data.get(
                        "value"
                    )
                )

                if value is None:
                    continue

                field = getattr(
                    self.profile,
                    field_name,
                )

                field.value = value

                field.confidence = (
                    field_data.get(
                        "confidence",
                        0.8,
                    )
                )

                field.evidence = (
                    field_data.get(
                        "evidence",
                        [],
                    )
                )

                accepted_fields[
                    field_name
                ] = {
                    "value": value,
                    "confidence": field.confidence,
                    "evidence": field.evidence,
                }

        # ----------------------------------------------------
        # 4. Determine missing fields.
        # ----------------------------------------------------

        missing_fields = (
            self.get_missing_fields()
        )

        ready = (
            len(missing_fields) == 0
        )

        # ----------------------------------------------------
        # 5. Knowledge search ONLY when complete.
        # ----------------------------------------------------

        if ready:

            try:

                self.last_knowledge_results = (
                    knowledge_searcher.search(
                        profile=self.profile_dict()
                    )
                )

            except Exception:

                self.last_knowledge_results = {}

        else:

            self.last_knowledge_results = {}

        # ----------------------------------------------------
        # 6. Build assistant response.
        # ----------------------------------------------------

        response = self.build_response(
            missing_fields
        )

        self.history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # ----------------------------------------------------
        # 7. Return structured analysis.
        # ----------------------------------------------------

        return ConversationAnalysis(

            extracted_fields=accepted_fields,

            relevant_information=[],

            irrelevant_information=[],

            ambiguous_information=[],

            missing_required_fields=missing_fields,

            fields_requiring_confirmation=[],

            next_question=(
                response
                if missing_fields
                else ""
            ),

            assistant_response=response,

            ready_for_recommendation=ready,
        )