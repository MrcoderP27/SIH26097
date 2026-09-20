from typing import Any, Dict, List, Set

from .data_store import data_store


class RecommendationEngine:
    """
    Transparent rule-based recommendation engine.

    The LLM is NOT used for recommendation ranking.

    Matching hierarchy:

        Occupation  -> strong signal
        Skills      -> strong signal
        Interests   -> supporting signal
        Aspirations -> supporting signal
        Employment  -> contextual livelihood signal
        Education   -> compatibility signal

    Important rule:

        Employment preference alone must NOT create a
        domain-specific recommendation.

    Example:

        "I want self-employment"

    must NOT automatically recommend:

        Food Processing
        Agriculture
        Solar Pump
        Cold Storage

    unless the beneficiary also has a relevant
    occupation, skill, interest, or aspiration.

    Education alone must never create a recommendation.
    """

    def __init__(self):
        self.data_store = data_store

    # =========================================================
    # BASIC TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_text(value: Any) -> str:

        if value is None:
            return ""

        if isinstance(value, dict):

            parts = []

            for item in value.values():

                normalized = RecommendationEngine.normalize_text(
                    item
                )

                if normalized:
                    parts.append(normalized)

            return " ".join(parts)

        if isinstance(value, list):

            parts = []

            for item in value:

                normalized = RecommendationEngine.normalize_text(
                    item
                )

                if normalized:
                    parts.append(normalized)

            return " ".join(parts)

        text = str(value).strip().lower()

        punctuation = [
            ",",
            ".",
            ":",
            ";",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "/",
            "\\",
            "|",
            "-",
            "_",
        ]

        for character in punctuation:
            text = text.replace(character, " ")

        return " ".join(text.split())

    # =========================================================
    # PROFILE VALUE EXTRACTION
    # =========================================================

    def get_profile_values(
        self,
        profile: Dict[str, Any],
        fields: List[str],
    ) -> List[str]:

        values = []

        for field_name in fields:

            field = profile.get(field_name)

            if field is None:
                continue

            if isinstance(field, dict):

                value = field.get("value")

            else:

                value = field

            if value is None:
                continue

            if isinstance(value, list):

                for item in value:

                    normalized = self.normalize_text(item)

                    if normalized:
                        values.append(normalized)

            else:

                normalized = self.normalize_text(value)

                if normalized:
                    values.append(normalized)

        return values

    # =========================================================
    # PROFILE GROUPS
    # =========================================================

    def build_profile_terms(
        self,
        profile: Dict[str, Any],
    ) -> Dict[str, List[str]]:

        return {

            "occupation": self.get_profile_values(
                profile,
                [
                    "current_occupation",
                    "traditional_occupation",
                ],
            ),

            "skills": self.get_profile_values(
                profile,
                [
                    "skills",
                ],
            ),

            "interests": self.get_profile_values(
                profile,
                [
                    "interests",
                ],
            ),

            "aspirations": self.get_profile_values(
                profile,
                [
                    "aspirations",
                ],
            ),

            "employment_preference": self.get_profile_values(
                profile,
                [
                    "employment_preference",
                ],
            ),

            "education": self.get_profile_values(
                profile,
                [
                    "education",
                ],
            ),
        }

    # =========================================================
    # SEARCH TERM EXPANSION
    # =========================================================

    @staticmethod
    def expand_terms(
        terms: List[str],
    ) -> List[str]:

        expanded: Set[str] = set()

        for original_term in terms:

            term = RecommendationEngine.normalize_text(
                original_term
            )

            if not term:
                continue

            expanded.add(term)

            # =================================================
            # VEHICLE / BIKE / AUTOMOBILE
            # =================================================

            vehicle_words = [
                "bike",
                "बाइक",
                "motorcycle",
                "motorbike",
                "मोटरसाइकिल",
                "मोटर साइकिल",
                "scooter",
                "स्कूटर",
                "vehicle",
                "वाहन",
                "automobile",
                "ऑटोमोबाइल",
            ]

            if any(word in term for word in vehicle_words):

                expanded.update(
                    [
                        "bike",
                        "bike repair",
                        "बाइक रिपेयर",
                        "बाइक की मरम्मत",
                        "motorcycle",
                        "motorcycle repair",
                        "motorbike",
                        "vehicle",
                        "vehicle repair",
                        "automobile",
                        "automobile repair",
                        "automobile servicing",
                        "engine repair",
                        "engine servicing",
                        "vehicle servicing",
                        "two wheeler",
                        "two wheeler repair",
                    ]
                )

            # =================================================
            # ENGINE / MECHANICAL
            # =================================================

            if any(
                word in term
                for word in [
                    "engine",
                    "इंजन",
                    "mechanic",
                    "मिस्त्री",
                    "mechanical",
                    "मैकेनिक",
                    "repair",
                    "मरम्मत",
                    "servicing",
                    "सर्विसिंग",
                ]
            ):

                expanded.update(
                    [
                        "engine",
                        "engine repair",
                        "engine servicing",
                        "mechanic",
                        "mechanical",
                        "repair",
                        "vehicle repair",
                        "automobile repair",
                        "automobile servicing",
                    ]
                )

            # =================================================
            # TAILORING / GARMENT
            # =================================================

            if any(
                word in term
                for word in [
                    "tailor",
                    "tailoring",
                    "tailor shop",
                    "silai",
                    "सिलाई",
                    "सिलाई का काम",
                    "दर्जी",
                    "garment",
                    "apparel",
                    "कपड़ा",
                ]
            ):

                expanded.update(
                    [
                        "tailor",
                        "tailoring",
                        "silai",
                        "सिलाई",
                        "दर्जी",
                        "garment",
                        "garment making",
                        "apparel",
                        "apparel making",
                        "dress making",
                    ]
                )

            # =================================================
            # AGRICULTURE / FARMING
            # =================================================

            if any(
                word in term
                for word in [
                    "agriculture",
                    "agricultural",
                    "farming",
                    "farmer",
                    "farm",
                    "खेती",
                    "किसान",
                    "कृषि",
                    "कृषक",
                ]
            ):

                expanded.update(
                    [
                        "agriculture",
                        "agricultural",
                        "farming",
                        "farmer",
                        "farm",
                        "खेती",
                        "किसान",
                        "कृषि",
                        "crop",
                    ]
                )

            # =================================================
            # WELDING
            # =================================================

            if any(
                word in term
                for word in [
                    "weld",
                    "welding",
                    "welder",
                    "वेल्डिंग",
                    "वेल्डर",
                ]
            ):

                expanded.update(
                    [
                        "welding",
                        "welder",
                        "weld",
                        "वेल्डिंग",
                        "वेल्डर",
                        "fabrication",
                        "metal fabrication",
                    ]
                )

            # =================================================
            # PLUMBING
            # =================================================

            if any(
                word in term
                for word in [
                    "plumb",
                    "plumber",
                    "plumbing",
                    "प्लंबर",
                    "प्लम्बिंग",
                    "नल",
                    "पाइप",
                    "पानी की पाइप",
                ]
            ):

                expanded.update(
                    [
                        "plumbing",
                        "plumber",
                        "pipe fitting",
                        "pipe",
                        "प्लंबर",
                        "प्लम्बिंग",
                        "पाइप फिटिंग",
                    ]
                )

            # =================================================
            # ELECTRICAL
            # =================================================

            if any(
                word in term
                for word in [
                    "electric",
                    "electrical",
                    "electrician",
                    "बिजली",
                    "इलेक्ट्रीशियन",
                    "इलेक्ट्रिकल",
                ]
            ):

                expanded.update(
                    [
                        "electrical",
                        "electrician",
                        "electric",
                        "electrical repair",
                        "wiring",
                        "electrical wiring",
                        "बिजली",
                        "इलेक्ट्रीशियन",
                        "वायरिंग",
                    ]
                )

            # =================================================
            # CONSTRUCTION / MASON
            # =================================================

            if any(
                word in term
                for word in [
                    "mason",
                    "masonry",
                    "construction",
                    "raj mistri",
                    "राजमिस्त्री",
                    "मिस्त्री",
                    "निर्माण",
                    "construction work",
                ]
            ):

                expanded.update(
                    [
                        "mason",
                        "masonry",
                        "construction",
                        "construction worker",
                        "building",
                        "raj mistri",
                        "राजमिस्त्री",
                        "निर्माण",
                    ]
                )

            # =================================================
            # FOOD / PROCESSING
            # =================================================

            if any(
                word in term
                for word in [
                    "food",
                    "food processing",
                    "pickle",
                    "pickles",
                    "papad",
                    "bakery",
                    "baking",
                    "cooking",
                    "खाना",
                    "खाद्य",
                    "खाद्य प्रसंस्करण",
                    "अचार",
                    "पापड़",
                    "बेकरी",
                    "बेकिंग",
                    "खाना बनाना",
                ]
            ):

                expanded.update(
                    [
                        "food",
                        "food processing",
                        "food industry",
                        "pickle",
                        "pickles",
                        "papad",
                        "bakery",
                        "baking",
                        "cooking",
                        "खाद्य प्रसंस्करण",
                        "अचार",
                        "पापड़",
                        "बेकरी",
                    ]
                )

            # =================================================
            # BEAUTY / PERSONAL CARE
            # =================================================

            if any(
                word in term
                for word in [
                    "beauty",
                    "beautician",
                    "salon",
                    "parlour",
                    "parlor",
                    "ब्यूटी",
                    "ब्यूटीशियन",
                    "पार्लर",
                    "सैलून",
                    "मेकअप",
                ]
            ):

                expanded.update(
                    [
                        "beauty",
                        "beautician",
                        "beauty parlour",
                        "beauty salon",
                        "salon",
                        "parlour",
                        "personal care",
                        "ब्यूटीशियन",
                        "ब्यूटी पार्लर",
                        "सैलून",
                        "मेकअप",
                    ]
                )

            # =================================================
            # DRIVING / TRANSPORT
            # =================================================

            if any(
                word in term
                for word in [
                    "driver",
                    "driving",
                    "drive",
                    "transport",
                    "वाहन चालक",
                    "ड्राइवर",
                    "ड्राइविंग",
                    "परिवहन",
                ]
            ):

                expanded.update(
                    [
                        "driver",
                        "driving",
                        "transport",
                        "commercial driver",
                        "vehicle driver",
                        "ड्राइवर",
                        "ड्राइविंग",
                        "परिवहन",
                    ]
                )

            # =================================================
            # MOBILE / COMPUTER / DIGITAL
            # =================================================

            if any(
                word in term
                for word in [
                    "mobile",
                    "मोबाइल",
                    "phone repair",
                    "mobile repair",
                    "computer",
                    "कंप्यूटर",
                    "laptop",
                    "digital",
                    "डिजिटल",
                ]
            ):

                expanded.update(
                    [
                        "mobile",
                        "mobile repair",
                        "phone repair",
                        "smartphone repair",
                        "computer",
                        "computer repair",
                        "laptop repair",
                        "digital",
                        "डिजिटल",
                        "मोबाइल रिपेयर",
                        "कंप्यूटर",
                    ]
                )

            # =================================================
            # DAIRY / LIVESTOCK
            # =================================================

            if any(
                word in term
                for word in [
                    "dairy",
                    "milk",
                    "livestock",
                    "cattle",
                    "goat",
                    "poultry",
                    "डेयरी",
                    "दूध",
                    "पशुपालन",
                    "बकरी",
                    "मुर्गी",
                    "पोल्ट्री",
                ]
            ):

                expanded.update(
                    [
                        "dairy",
                        "milk",
                        "livestock",
                        "cattle",
                        "goat",
                        "poultry",
                        "animal husbandry",
                        "डेयरी",
                        "दूध",
                        "पशुपालन",
                        "पोल्ट्री",
                    ]
                )

            # =================================================
            # HANDICRAFT / ARTISAN
            # =================================================

            if any(
                word in term
                for word in [
                    "handicraft",
                    "handloom",
                    "artisan",
                    "craft",
                    "हस्तशिल्प",
                    "हस्तकरघा",
                    "कारीगर",
                    "शिल्प",
                ]
            ):

                expanded.update(
                    [
                        "handicraft",
                        "handloom",
                        "artisan",
                        "craft",
                        "हस्तशिल्प",
                        "हस्तकरघा",
                        "कारीगर",
                        "शिल्प",
                    ]
                )

            # =================================================
            # BUSINESS / SELF EMPLOYMENT
            #
            # IMPORTANT:
            #
            # We intentionally DO NOT expand:
            #
            # business -> entrepreneur
            # enterprise -> entrepreneur
            #
            # because those generic words create false
            # matches against unrelated schemes.
            # =================================================

            if any(
                word in term
                for word in [
                    "self employment",
                    "self-employment",
                    "own business",
                    "own work",
                    "अपना काम",
                    "अपना व्यवसाय",
                    "स्वरोजगार",
                    "self employed",
                    "स्वयं रोजगार",
                ]
            ):

                expanded.update(
                    [
                        "self employment",
                        "self-employment",
                        "self employed",
                        "own business",
                        "own work",
                        "अपना काम",
                        "अपना व्यवसाय",
                        "स्वरोजगार",
                        "स्वयं रोजगार",
                    ]
                )

            if any(
                word in term
                for word in [
                    "business",
                    "व्यवसाय",
                    "बिजनेस",
                    "व्यापार",
                ]
            ):

                expanded.update(
                    [
                        "business",
                        "व्यवसाय",
                        "बिजनेस",
                        "व्यापार",
                    ]
                )

        return sorted(expanded)

    # =========================================================
    # EMPLOYMENT PREFERENCE
    # =========================================================

    @staticmethod
    def is_self_employment_preference(
        employment_values: List[str],
    ) -> bool:

        text = " ".join(employment_values)

        self_employment_terms = [
            "self employment",
            "self-employment",
            "self employed",
            "own business",
            "own work",
            "business",
            "व्यवसाय",
            "बिजनेस",
            "व्यापार",
            "अपना काम",
            "अपना व्यवसाय",
            "स्वरोजगार",
            "स्वयं रोजगार",
        ]

        return any(
            term in text
            for term in self_employment_terms
        )

    @staticmethod
    def is_wage_employment_preference(
        employment_values: List[str],
    ) -> bool:

        text = " ".join(employment_values)

        wage_terms = [
            "wage employment",
            "wage-employment",
            "job",
            "नौकरी",
            "रोजगार",
        ]

        return any(
            term in text
            for term in wage_terms
        )

    # =========================================================
    # GENERAL SELF-EMPLOYMENT SUPPORT DETECTION
    # =========================================================

    def is_general_self_employment_opportunity(
        self,
        record: Dict[str, Any],
    ) -> bool:

        """
        Detect whether a scheme itself is broadly designed
        for self-employment / livelihood / enterprise support.

        This is deliberately conservative.

        A record such as:

            PMEGP - Self Employment / New Enterprise

        should qualify.

        A record such as:

            PMFME - Micro Food Processing Unit

        should NOT qualify merely because its target
        beneficiary contains "entrepreneurs".
        """

        title = self.normalize_text(
            record.get("job_title")
        )

        required_skill = self.normalize_text(
            record.get("required_skill")
        )

        target = self.normalize_text(
            record.get("target_beneficiary")
        )

        combined = " ".join(
            [
                title,
                required_skill,
                target,
            ]
        )

        explicit_general_terms = [
            "self employment",
            "self-employment",
            "micro enterprise",
            "micro-enterprise",
            "rural enterprise",
            "rural business",
            "income generation",
            "income-generating",
            "livelihood",
            "livelihood enhancement",
            "economic development",
            "enterprise support",
            "enterprise promotion",
            "new enterprise",
            "own business",
        ]

        return any(
            term in combined
            for term in explicit_general_terms
        )

    # =========================================================
    # EDUCATION COMPATIBILITY
    # =========================================================

    @staticmethod
    def education_is_compatible(
        profile_education: List[str],
        eligibility: str,
    ) -> bool:

        education_text = " ".join(
            profile_education
        ).lower()

        eligibility_text = (
            eligibility or ""
        ).lower()

        if not education_text:
            return False

        # -----------------------------------------------------
        # 10TH
        # -----------------------------------------------------

        if any(
            value in education_text
            for value in [
                "10th",
                "10वीं",
                "दसवीं",
                "class 10",
            ]
        ):

            if any(
                value in eligibility_text
                for value in [
                    "10th",
                    "10वीं",
                    "दसवीं",
                    "class 10",
                ]
            ):

                return True

        # -----------------------------------------------------
        # 12TH
        # -----------------------------------------------------

        if any(
            value in education_text
            for value in [
                "12th",
                "12वीं",
                "बारहवीं",
                "class 12",
            ]
        ):

            if any(
                value in eligibility_text
                for value in [
                    "10th",
                    "12th",
                    "10वीं",
                    "12वीं",
                    "दसवीं",
                    "बारहवीं",
                    "class 10",
                    "class 12",
                ]
            ):

                return True

        return False

    # =========================================================
    # QUALIFICATION SEARCH TEXT
    # =========================================================

    def qualification_search_text(
        self,
        record: Dict[str, Any],
    ) -> str:

        fields = [
            "title",
            "sector",
            "eligibility",
            "required_skill",
            "skills",
            "description",
            "objective",
            "outcome",
            "occupation",
            "trade",
            "job_title",
        ]

        parts = []

        for field in fields:

            value = record.get(field)

            normalized = self.normalize_text(
                value
            )

            if normalized:
                parts.append(normalized)

        return " ".join(parts)

    # =========================================================
    # JOB / SCHEME SEARCH TEXT
    # =========================================================

    def job_search_text(
        self,
        record: Dict[str, Any],
    ) -> str:

        fields = [
            "job_title",
            "title",
            "required_skill",
            "skills",
            "target_beneficiary",
            "support",
            "description",
            "objective",
            "occupation",
            "sector",
            "eligibility",
            "trade",
        ]

        parts = []

        for field in fields:

            value = record.get(field)

            normalized = self.normalize_text(
                value
            )

            if normalized:
                parts.append(normalized)

        return " ".join(parts)

    # =========================================================
    # MATCH TERMS
    # =========================================================

    def find_matches(
        self,
        terms: List[str],
        record_text: str,
    ) -> List[str]:

        matches: Set[str] = set()

        normalized_record = self.normalize_text(
            record_text
        )

        if not normalized_record:
            return []

        expanded_terms = self.expand_terms(
            terms
        )

        for term in expanded_terms:

            normalized_term = self.normalize_text(
                term
            )

            if not normalized_term:
                continue

            # -------------------------------------------------
            # Exact phrase match
            # -------------------------------------------------

            if normalized_term in normalized_record:

                matches.add(normalized_term)

                continue

            # -------------------------------------------------
            # Multi-word fallback
            #
            # Require ALL meaningful words.
            # This is stricter than the previous
            # "two words are enough" rule.
            # -------------------------------------------------

            words = [
                word
                for word in normalized_term.split()
                if len(word) >= 3
            ]

            if len(words) >= 2:

                if all(
                    word in normalized_record
                    for word in words
                ):

                    matches.add(
                        normalized_term
                    )

        return sorted(matches)

    # =========================================================
    # QUALIFICATION MATCHING
    # =========================================================

    def score_qualification(
        self,
        record: Dict[str, Any],
        profile_groups: Dict[str, List[str]],
    ) -> Dict[str, Any]:

        if not isinstance(record, dict):

            return {
                "score": 0,
                "matched_signals": [],
                "reasons": [],
                "record": record,
            }

        record_text = self.qualification_search_text(
            record
        )

        score = 0

        matched_signals = []

        reasons = []

        # -----------------------------------------------------
        # OCCUPATION
        # -----------------------------------------------------

        occupation_matches = self.find_matches(
            profile_groups.get(
                "occupation",
                [],
            ),
            record_text,
        )

        if occupation_matches:

            score += 5

            matched_signals.append(
                {
                    "type": "occupation",
                    "matched_terms": occupation_matches,
                }
            )

            reasons.append(
                "यह प्रशिक्षण लाभार्थी के वर्तमान या पारंपरिक व्यवसाय से संबंधित है।"
            )

        # -----------------------------------------------------
        # SKILLS
        # -----------------------------------------------------

        skill_matches = self.find_matches(
            profile_groups.get(
                "skills",
                [],
            ),
            record_text,
        )

        if skill_matches:

            score += 5

            matched_signals.append(
                {
                    "type": "skills",
                    "matched_terms": skill_matches,
                }
            )

            reasons.append(
                "यह प्रशिक्षण लाभार्थी के वर्तमान बताई गई कौशल से संबंधित है।" 
            )

        # -----------------------------------------------------
        # INTERESTS
        # -----------------------------------------------------

        interest_matches = self.find_matches(
            profile_groups.get(
                "interests",
                [],
            ),
            record_text,
        )

        if interest_matches:

            score += 3

            matched_signals.append(
                {
                    "type": "interests",
                    "matched_terms": interest_matches,
                }
            )

            reasons.append(
                "यह प्रशिक्षण लाभार्थी के वर्तमान या पारंपरिक बताई गई रुचियों से संबंधित है।"
            )

        # -----------------------------------------------------
        # ASPIRATIONS
        # -----------------------------------------------------

        aspiration_matches = self.find_matches(
            profile_groups.get(
                "aspirations",
                [],
            ),
            record_text,
        )

        if aspiration_matches:

            score += 2

            matched_signals.append(
                {
                    "type": "aspirations",
                    "matched_terms": aspiration_matches,
                }
            )

            reasons.append(
                "यह प्रशिक्षण लाभार्थी के वर्तमान आकांक्षा से संबंधित है।"
            )

        # -----------------------------------------------------
        # EDUCATION
        # -----------------------------------------------------

        eligibility = self.normalize_text(
            record.get("eligibility")
        )

        if self.education_is_compatible(
            profile_groups.get(
                "education",
                [],
            ),
            eligibility,
        ):

            matched_signals.append(
                {
                    "type": "education",
                    "matched_terms": profile_groups.get(
                        "education",
                        [],
                    ),
                }
            )

            reasons.append(
                "Education appears compatible with "
                "the listed eligibility."
            )

        # -----------------------------------------------------
        # EDUCATION ALONE NEVER RECOMMENDS
        # -----------------------------------------------------

        if score == 0:

            return {
                "score": 0,
                "matched_signals": [],
                "reasons": [],
                "record": record,
            }

        return {
            "score": score,
            "matched_signals": matched_signals,
            "reasons": reasons,
            "record": record,
        }

    # =========================================================
    # JOB / SCHEME MATCHING
    # =========================================================

    def score_job(
        self,
        record: Dict[str, Any],
        profile_groups: Dict[str, List[str]],
    ) -> Dict[str, Any]:

        if not isinstance(record, dict):

            return {
                "score": 0,
                "matched_signals": [],
                "reasons": [],
                "record": record,
            }

        record_text = self.job_search_text(
            record
        )

        score = 0

        matched_signals = []

        reasons = []

        # -----------------------------------------------------
        # OCCUPATION
        # -----------------------------------------------------

        occupation_matches = self.find_matches(
            profile_groups.get(
                "occupation",
                [],
            ),
            record_text,
        )

        if occupation_matches:

            score += 5

            matched_signals.append(
                {
                    "type": "occupation",
                    "matched_terms": occupation_matches,
                }
            )

            reasons.append(
                "यह अवसर लाभार्थी के व्यवसाय से संबंधित है।"
            )

        # -----------------------------------------------------
        # SKILLS
        # -----------------------------------------------------

        skill_matches = self.find_matches(
            profile_groups.get(
                "skills",
                [],
            ),
            record_text,
        )

        if skill_matches:

            score += 5

            matched_signals.append(
                {
                    "type": "skills",
                    "matched_terms": skill_matches,
                }
            )

            reasons.append(
                "यह अवसर लाभार्थी के कौशल से संबंधित है।"
            )

        # -----------------------------------------------------
        # INTERESTS
        # -----------------------------------------------------

        interest_matches = self.find_matches(
            profile_groups.get(
                "interests",
                [],
            ),
            record_text,
        )

        if interest_matches:

            score += 3

            matched_signals.append(
                {
                    "type": "interests",
                    "matched_terms": interest_matches,
                }
            )

            reasons.append(
                "यह अवसर लाभार्थी के रुचियों से संबंधित है।"
            )

        # -----------------------------------------------------
        # ASPIRATIONS
        # -----------------------------------------------------

        aspiration_matches = self.find_matches(
            profile_groups.get(
                "aspirations",
                [],
            ),
            record_text,
        )

        if aspiration_matches:

            score += 2

            matched_signals.append(
                {
                    "type": "aspirations",
                    "matched_terms": aspiration_matches,
                }
            )

            reasons.append(
                "यह अवसर लाभार्थी के आशा से संबंधित है।"
            )

        # -----------------------------------------------------
        # EMPLOYMENT PREFERENCE
        # -----------------------------------------------------

        employment_matches = self.find_matches(
            profile_groups.get(
                "employment_preference",
                [],
            ),
            record_text,
        )

        # -----------------------------------------------------
        # CRITICAL RULE
        #
        # Employment preference alone must NOT recommend
        # a domain-specific opportunity.
        # -----------------------------------------------------

        domain_signal_exists = any(
            [
                bool(occupation_matches),
                bool(skill_matches),
                bool(interest_matches),
                bool(aspiration_matches),
            ]
        )

        if employment_matches:

            if domain_signal_exists:

                score += 4

                matched_signals.append(
                    {
                        "type": "employment_preference",
                        "matched_terms": employment_matches,
                    }
                )

                reasons.append(
                    "यह अवसर लाभार्थी के रोजगार प्राधार से संबंधित है।"
                )

            else:

                # Only allow employment preference alone
                # for schemes that are explicitly general
                # self-employment / livelihood support.
                if (
                    self.is_self_employment_preference(
                        profile_groups.get(
                            "employment_preference",
                            [],
                        )
                    )
                    and self.is_general_self_employment_opportunity(
                        record
                    )
                ):

                    score += 4

                    matched_signals.append(
                        {
                            "type": "employment_preference",
                            "matched_terms": employment_matches,
                        }
                    )

                    reasons.append(
                        "The opportunity is a general "
                        "self-employment or livelihood "
                        "support pathway."
                    )

        # -----------------------------------------------------
        # NO MATCH
        # -----------------------------------------------------

        if score == 0:

            return {
                "score": 0,
                "matched_signals": [],
                "reasons": [],
                "record": record,
            }

        return {
            "score": score,
            "matched_signals": matched_signals,
            "reasons": reasons,
            "record": record,
        }

    # =========================================================
    # MAIN RECOMMENDATION
    # =========================================================

    def recommend(
        self,
        profile: Dict[str, Any],
        limit: int = 5,
    ) -> Dict[str, Any]:

        if not isinstance(profile, dict):

            return {
                "profile_groups": {},
                "profile_terms": [],
                "expanded_search_terms": [],
                "qualification_matches": [],
                "job_or_scheme_matches": [],
            }

        profile_groups = self.build_profile_terms(
            profile
        )

        # -----------------------------------------------------
        # ORIGINAL PROFILE TERMS
        # -----------------------------------------------------

        profile_terms = []

        seen_profile_terms = set()

        for values in profile_groups.values():

            for value in values:

                if value not in seen_profile_terms:

                    profile_terms.append(value)

                    seen_profile_terms.add(value)

        # -----------------------------------------------------
        # EXPANDED TERMS
        # -----------------------------------------------------

        expanded_terms = self.expand_terms(
            profile_terms
        )

        # -----------------------------------------------------
        # QUALIFICATIONS
        # -----------------------------------------------------

        qualifications = (
            self.data_store.get_qualifications()
        )

        qualification_matches = []

        for record in qualifications:

            result = self.score_qualification(
                record,
                profile_groups,
            )

            if result["score"] > 0:

                qualification_matches.append(
                    result
                )

        qualification_matches.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # -----------------------------------------------------
        # JOBS / SCHEMES
        # -----------------------------------------------------

        jobs = self.data_store.get_jobs()

        job_matches = []

        for record in jobs:

            result = self.score_job(
                record,
                profile_groups,
            )

            if result["score"] > 0:

                job_matches.append(
                    result
                )

        job_matches.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # -----------------------------------------------------
        # FINAL RESPONSE
        # -----------------------------------------------------

        return {

            "profile_groups": profile_groups,

            "profile_terms": profile_terms,

            "expanded_search_terms": expanded_terms,

            "qualification_matches": (
                qualification_matches[:limit]
            ),

            "job_or_scheme_matches": (
                job_matches[:limit]
            ),
        }


recommendation_engine = RecommendationEngine()