import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from database import create_connection, close_connection

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해 주세요.")

def get_chatbot_response(user_message: str) -> str:
    # 데이터베이스 연결 시도
    connection = create_connection()
    if not connection:
        print("데이터베이스 연결에 실패했습니다. GPT로부터 응답을 생성합니다.")
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
        # ChatOpenAI 모델 인스턴스 생성, API 키를 명시적으로 전달
        chat_model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=150, openai_api_key=openai_api_key)

        # GPT 모델을 사용하여 응답 생성
        response = chat_model([HumanMessage(content=user_message)])

        # 응답 텍스트 반환
        return response.content
    except Exception as e:
        return f"GPT 호출 오류 발생: {e}"





