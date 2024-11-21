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

def moderate_content(content: str) -> bool:
    """Moderates the content using OpenAI's Moderation API.
    
    Returns True if the content is flagged as inappropriate.
    """
    try:
        moderation_response = openai.Moderation.create(
            input=content,
            api_key=openai_api_key  # 명시적으로 API 키 전달
        )
        flagged = moderation_response["results"][0]["flagged"]
        return flagged
    except Exception as e:
        print(f"Moderation API 호출 오류 발생: {e}")
        return False

# 정치적 키워드 필터링 함수
def filter_political_content(response: str) -> str:
    political_keywords = [
        "정치", "대통령", "정당", "정권", "정부", "선거", "국회의원", "민주주의", "외교", "국방"
    ]
    if any(keyword in response for keyword in political_keywords):
        return "해당 주제는 답변드릴 수 없습니다. 진로 관련 질문을 해주세요."
    return response

def get_chatbot_response(user_message: str) -> str:
    connection = create_connection()
    if not connection:
        print("데이터베이스 연결에 실패했습니다. GPT로부터 응답을 생성합니다.")
        return process_gpt_response(user_message, generate_gpt_response(user_message))

    try:
        cursor = connection.cursor()
        query = "SELECT response FROM chatbot_responses WHERE message = %s"
        cursor.execute(query, (user_message,))
        result = cursor.fetchone()

        if result:
            return process_gpt_response(user_message, result[0])

        # 데이터베이스에 응답이 없는 경우 GPT로 응답 생성
        response = generate_gpt_response(user_message)

        # 응답을 검열
        if moderate_content(response):
            return "죄송합니다, 적절하지 않은 응답이 생성되었습니다. 다른 질문을 해주세요."

        return process_gpt_response(user_message, response)

    except Exception as e:
        print(f"데이터베이스 오류 발생: {e}")
        return process_gpt_response(user_message, generate_gpt_response(user_message))

    finally:
        close_connection(connection)

def generate_gpt_response(user_message: str) -> str:
    try:
        # 프롬프트에 진로 상담 관련 지시사항 추가
        prompt = (
            "당신은 대학생의 진로 상담을 돕는 AI 챗봇입니다.\n"
            "사용자의 질문에 공감하며 실용적이고 구체적인 진로 조언을 제공하세요.\n"
            "정치적, 종교적, 논쟁적인 주제는 다루지 않습니다.\n"
            "질문에 공감적이고 친절한 어조로 답변하세요.\n"
            "예시:\n"
            "- 전공이 잘 맞지 않는다고 느끼시는군요. 많은 학생들이 그런 고민을 하고 있으니 너무 걱정하지 않으셔도 돼요.\n"
            "- 현재는 확신이 서지 않을 수 있지만, 다양한 경험을 통해 분명히 자신에게 맞는 길을 찾을 수 있을 거예요.\n"
            "- 마케팅에 관심이 있으시다면, '소셜 미디어 브랜딩' 같은 수업을 들어보세요.\n"
            f"학생 질문: {user_message}\n답변:"
        )
        
        
        # ChatOpenAI 모델 인스턴스 생성, API 키를 명시적으로 전달
        chat_model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=200, temperature= 0.3, openai_api_key=openai_api_key)

        # GPT 모델을 사용하여 응답 생성
        response = chat_model([HumanMessage(content=prompt)])

        # 응답 텍스트 반환
        return response.content
    except Exception as e:
        return f"GPT 호출 오류 발생: {e}"

def process_gpt_response(user_message: str, response: str) -> str:
    filtered_response = filter_political_content(response)
    return filtered_response
