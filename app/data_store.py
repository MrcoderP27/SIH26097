import json
from pathlib import Path
from typing import Any, Dict, List


# Project root:
# pm_ajay_ai/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data.json"

class DataStore:
    """
    Loads and provides access to the prototype PM-AJAY knowledge data.

    Current prototype source:
        data.json

    Later this class can be changed to read:
        PostgreSQL / Firebase / another database

    without changing the AI/recommendation logic.
    """

    def __init__(self, data_file: Path = DATA_FILE):
        self.data_file = data_file
        self.data: Dict[str, Any] = {}

        self.load()

    def load(self) -> None:
        """Load data.json into memory."""

        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_file}"
            )

        try:
            with open(
                self.data_file,
                "r",
                encoding="utf-8"
            ) as file:

                self.data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {self.data_file}: {exc}"
            ) from exc

    def reload(self) -> None:
        """Reload data from disk."""

        self.load()

    def _find_list_by_key(self, possible_keys: List[str]) -> List[Dict[str, Any]]:
        """
        Find a list in the JSON using case-insensitive key matching.

        This allows small differences such as:
            qualifications
            Qualifications
            qualification
            jobs
            Jobs
        """

        if not isinstance(self.data, dict):
            return []

        normalized_keys = {
            str(key).lower().strip(): key
            for key in self.data.keys()
        }

        for possible_key in possible_keys:

            normalized = possible_key.lower().strip()

            if normalized in normalized_keys:

                original_key = normalized_keys[normalized]

                value = self.data[original_key]

                if isinstance(value, list):
                    return value

        return []

    def get_qualifications(self) -> List[Dict[str, Any]]:
        """Return qualification records."""

        return self._find_list_by_key(
            [
                "qualifications",
                "qualification",
                "courses",
                "courses_and_qualifications"
            ]
        )

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Return job records."""

        return self._find_list_by_key(
            [
                "jobs",
                "job",
                "job_opportunities",
                "employment"
            ]
        )

    def get_all_data(self) -> Dict[str, Any]:
        """Return the complete loaded dataset."""

        return self.data

    def summary(self) -> Dict[str, Any]:
        """Return a small summary useful for testing."""

        qualifications = self.get_qualifications()
        jobs = self.get_jobs()

        return {
            "data_file": str(self.data_file),
            "qualifications_count": len(qualifications),
            "jobs_count": len(jobs),
            "top_level_keys": list(self.data.keys())
            if isinstance(self.data, dict)
            else []
        }


# Shared data store instance
data_store = DataStore()