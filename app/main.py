from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .conversation import ConversationEngine
from .data_store import data_store
from .knowledge_search import knowledge_searcher
from .recommendation_engine import recommendation_engine

app = FastAPI(
    title="PM-AJAY AI Conversation Engine",
    version="0.1.0"
)


engines = {}


class MessageRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def root():
    return {
        "project": "PM-AJAY AI Livelihood Assistant",
        "status": "running"
    }


@app.get("/data-summary")
def data_summary():
    return data_store.summary()

@app.post("/search-knowledge")
def search_knowledge(profile: dict):

    return knowledge_searcher.search(
        profile=profile
    )


@app.post("/chat")
def chat(request: MessageRequest):
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # Create conversation engine for new session
        if request.session_id not in engines:
            engines[request.session_id] = ConversationEngine()

        engine = engines[request.session_id]

        # Process the user's message
        result = engine.process_message(request.message)

        # Get current structured beneficiary profile
        profile = engine.profile_dict()

        # Get currently missing fields
        missing_fields = engine.get_missing_fields()

        # Default recommendation response
        recommendations = {
            "qualification_matches": [],
            "job_or_scheme_matches": []
        }

        # Only recommend when the conversation has enough information
        if result.ready_for_recommendation:
            recommendations = recommendation_engine.recommend(
                profile=profile,
                limit=5
            )

        return {
            "session_id": request.session_id,

            "assistant": result.assistant_response,

            "next_question": result.next_question,

            "analysis": result.model_dump(),

            "profile": profile,

            "missing_fields": missing_fields,

            "knowledge_results": engine.last_knowledge_results,

            "recommendations": recommendations
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

@app.post("/recommend")
def recommend(profile: dict):
    try:
        return recommendation_engine.recommend(
            profile=profile,
            limit=5,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
    