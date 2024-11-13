from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import get_chatbot_response  # 챗봇 로직 불러오기

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "welcom to the FastAPI application!"}

@app.post("/chat/")
async def chat(request: ChatRequest):
    user_message = request.message
    try:
        # get_chatbot_response 함수 호출하여 응답 생성
        response = get_chatbot_response(user_message)
        return {"response": response}
    except Exception as e:
        # 오류 발생 시 HTTPException 반환
        raise HTTPException(status_code=500, detail=str(e))