from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import initialize_rag_resources, get_chatbot_response

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    # RAG 리소스 초기화
    initialize_rag_resources()

@app.post("/chat/")
async def chat(request: ChatRequest):
    user_message = request.message
    try:
        # RAG와 GPT 통합 응답 생성
        response = get_chatbot_response(user_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


