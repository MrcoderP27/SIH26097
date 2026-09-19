from typing import Any, Dict, List

from .data_store import data_store


class KnowledgeSearcher:
    """
    Searches the verified prototype knowledge data.

    This component does NOT use the LLM.

    Its job is to find records that are relevant to a beneficiary profile.
    """

    def __init__(self):
        self.data_store = data_store

    @staticmethod
    def _text(value: Any) -> str:
        """
        Convert any JSON value into searchable lowercase text.
        """

        if value is None:
            return ""

        if isinstance(value, list):
            return " ".join(
                KnowledgeSearcher._text(item)
                for item in value
            ).lower()

        if isinstance(value, dict):
            return " ".join(
                KnowledgeSearcher._text(item)
                for item in value.values()
            ).lower()

        return str(value).lower()

    @staticmethod
    def _profile_terms(profile: Dict[str, Any]) -> List[str]:
        """
        Extract useful searchable terms from the beneficiary profile.
        """

        terms = []

        for field_name in [
            "current_occupation",
            "traditional_occupation",
            "skills",
            "interests",
            "aspirations",
            "education",
        ]:

            field = profile.get(field_name)

            if not isinstance(field, dict):
                continue

            value = field.get("value")

            if value is None:
                continue

            if isinstance(value, list):

                for item in value:

                    if item:
                        terms.append(
                            str(item).lower().strip()
                        )

            else:

                terms.append(
                    str(value).lower().strip()
                )

        return [
            term
            for term in terms
            if term
        ]

    @staticmethod
    def _score_record(
        record: Dict[str, Any],
        terms: List[str]
    ) -> int:
        """
        Calculate a simple transparent relevance score.

        This is intentionally deterministic.

        The LLM does not decide the score.
        """

        if not isinstance(record, dict):
            return 0

        record_text = KnowledgeSearcher._text(record)

        score = 0

        for term in terms:

            if term in record_text:
                score += 1

        return score

    def search(
        self,
        profile: Dict[str, Any],
        limit: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search qualifications and jobs against the beneficiary profile.
        """

        terms = self._profile_terms(profile)

        qualifications = (
            self.data_store.get_qualifications()
        )

        jobs = (
            self.data_store.get_jobs()
        )

        scored_qualifications = []

        for record in qualifications:

            score = self._score_record(
                record,
                terms
            )

            if score > 0:

                scored_qualifications.append(
                    {
                        "score": score,
                        "record": record
                    }
                )

        scored_jobs = []

        for record in jobs:

            score = self._score_record(
                record,
                terms
            )

            if score > 0:

                scored_jobs.append(
                    {
                        "score": score,
                        "record": record
                    }
                )

        scored_qualifications.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        scored_jobs.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return {
            "search_terms": terms,

            "qualifications": scored_qualifications[:limit],

            "jobs": scored_jobs[:limit]
        }


knowledge_searcher = KnowledgeSearcher()