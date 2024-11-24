import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from database import create_connection, close_connection

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해 주세요.")

DB_PATHS = ["./vectordb/test", "./vectordb/certification"]  # 벡터 DB 경로
MODEL_NAME = "BAAI/bge-large-en-v1.5"
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

def load_and_merge_vectorstores(db_paths, embedding_model):
    vectorstore = None
    for db_path in db_paths:
        temp_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_serialization=True)
        if vectorstore is None:
            vectorstore = temp_store
        else:
            vectorstore.merge_from(temp_store)
    return vectorstore

def get_rag_response(user_message: str) -> str:
    vectorstore, openai_model = initialize_rag_resources()
    try:
        system_prompt_template = apply_prompt_template()
        qa = ConversationalRetrievalChain.from_llm(
            llm=openai_model,
            retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True
        )
        response = qa.run(user_message)
        return response
    except Exception as e:
        raise ValueError(f"Error during RAG processing: {str(e)}")

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
def filter_political_message(user_message: str) -> str:
    political_keywords = [
        "정치", "정당", "정권", "정부", "선거", "민주주의", "외교", "국방"
    ]
    if any(keyword in user_message for keyword in political_keywords):
        return False
    return user_message

def get_chatbot_response(user_message: str) -> str:
    connection = create_connection()
    if not connection:
        print("데이터베이스 연결에 실패했습니다. RAG 응답을 생성합니다.")
        return get_rag_response(user_message)

    try:
        cursor = connection.cursor()
        query = "SELECT response FROM chatbot_responses WHERE message = %s"
        cursor.execute(query, (user_message,))
        result = cursor.fetchone()

        if result:
            return result[0]

        # 데이터베이스에 응답이 없는 경우 RAG 응답 생성
        if filter_political_message(user_message):
            user_message = filter_political_message(user_message)
        else:
            return "해당 주제는 답변드릴 수 없습니다. 자격증 관련한 질문 해주세요."

        # 응답을 검열
        if moderate_content(user_message):
            return "죄송합니다, 적절하지 않은 응답이 생성되었습니다. 다른 질문을 해주세요."
        
        response = get_rag_response(user_message)

        return response

    except Exception as e:
        print(f"데이터베이스 오류 발생: {e}")
        return get_rag_response(user_message)

    finally:
        close_connection(connection)


def initialize_rag_resources():
    global vectorstore, openai_model
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = load_and_merge_vectorstores(DB_PATHS, embeddings)

    openai_model = ChatOpenAI(
        model_name=OPENAI_MODEL,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.3,
        max_tokens=200,
        openai_api_key=openai_api_key
    )
    return vectorstore,openai_model


def apply_prompt_template():
    system_prompt_template = PromptTemplate(
        input_variables=["infos", "user_message"],
        template="""" "당신은 대학생의 진로 상담을 돕는 AI 챗봇입니다.\n"
            "사용자의 질문에 공감하며 실용적이고 구체적인 진로 조언을 제공하세요.\n"
            "정치적, 종교적, 논쟁적인 주제는 다루지 않습니다.\n"
            "질문에 공감적이고 친절한 어조로 답변하세요.\n"
            "예시:\n"
            "- 전공이 잘 맞지 않는다고 느끼시는군요. 많은 학생들이 그런 고민을 하고 있으니 너무 걱정하지 않으셔도 돼요.\n"
            "- 현재는 확신이 서지 않을 수 있지만, 다양한 경험을 통해 분명히 자신에게 맞는 길을 찾을 수 있을 거예요.\n"
            "- 마케팅에 관심이 있으시다면, '소셜 미디어 브랜딩' 같은 수업을 들어보세요.\n"
            f"학생 질문: {user_message}\n답변:"
        """)
    return system_prompt_template
