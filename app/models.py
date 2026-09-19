from typing import Any, Dict, List, Optional
# Pydantic is an optional runtime dependency in some environments.  The type
# ignore keeps editors from reporting a missing package while deployments that
# use these models can still install and provide Pydantic normally.
from pydantic import BaseModel, Field  # type: ignore[import-not-found]


class FieldEvidence(BaseModel):
    value: Any = None
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class BeneficiaryProfile(BaseModel):
    name: FieldEvidence = Field(default_factory=FieldEvidence)
    age: FieldEvidence = Field(default_factory=FieldEvidence)
    education: FieldEvidence = Field(default_factory=FieldEvidence)

    current_occupation: FieldEvidence = Field(default_factory=FieldEvidence)
    traditional_occupation: FieldEvidence = Field(default_factory=FieldEvidence)

    experience: FieldEvidence = Field(default_factory=FieldEvidence)

    skills: FieldEvidence = Field(default_factory=FieldEvidence)
    interests: FieldEvidence = Field(default_factory=FieldEvidence)
    aspirations: FieldEvidence = Field(default_factory=FieldEvidence)

    income_goal: FieldEvidence = Field(default_factory=FieldEvidence)

    employment_preference: FieldEvidence = Field(default_factory=FieldEvidence)

    mobility: FieldEvidence = Field(default_factory=FieldEvidence)
    work_constraints: FieldEvidence = Field(default_factory=FieldEvidence)

    location: FieldEvidence = Field(default_factory=FieldEvidence)
    language: FieldEvidence = Field(default_factory=FieldEvidence)

    previous_training: FieldEvidence = Field(default_factory=FieldEvidence)


class ConversationAnalysis(BaseModel):
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)

    relevant_information: List[str] = Field(default_factory=list)

    irrelevant_information: List[str] = Field(default_factory=list)

    ambiguous_information: List[str] = Field(default_factory=list)

    missing_required_fields: List[str] = Field(default_factory=list)

    fields_requiring_confirmation: List[str] = Field(default_factory=list)

    next_question: str = ""

    assistant_response: str = ""

    ready_for_recommendation: bool = False