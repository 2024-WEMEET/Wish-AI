import openai
import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from database import create_connection, close_connection

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("sk-svcacct-IQFqO84FdCKXC8s9ZJikTypIAVnRknNseMZVdhjxVwKw1b7v-7hNWsfRHw1YYiT3BlbkFJ6IhVPGelfVBzlliX6tZlasIfcRQJWdJpdZA2VUpuSAklBDBcGnT_BrTaJmFGEA")

def get_chatbot_response(user_message: str) -> str:
    # 데이터베이스 연결 시도
    connection = create_connection()
    if not connection:
        print("데이터베이스 연결에 실패했습니다. GPT로부터 응답을 생성합니다.")
        # 데이터베이스 연결 실패 시 GPT 응답 생성
        return generate_gpt_response(user_message)

    try:
        # 데이터베이스에서 응답 조회
        cursor = connection.cursor()
        query = "SELECT response FROM chatbot_responses WHERE message = %s"
        cursor.execute(query, (user_message,))
        result = cursor.fetchone()

        # 데이터베이스에 응답이 있는 경우 해당 응답 반환
        if result:
            return result[0]
        
        # 데이터베이스에 응답이 없는 경우 GPT로 응답 생성
        return generate_gpt_response(user_message)
    
    except Exception as e:
        print(f"데이터베이스 오류 발생: {e}")
        return generate_gpt_response(user_message)
    
    finally:
        close_connection(connection)

def generate_gpt_response(user_message: str) -> str:
    try:
        # 최신 OpenAI GPT 모델 호출
        response = ChatOpenAI(
            model="gpt-3.5-turbo",  # 또는 "gpt-4" 사용 가능
            messages=[
                {"role": "user", "content": user_message}
            ],
            max_tokens=150  # 필요한 최대 토큰 수를 설정
        )
        # GPT 응답 텍스트 추출
        return response.choices[0].message['content']
    except Exception as e:
        return f"GPT 호출 오류 발생: {e}"



