import os
from database import create_connection, close_connection, get_recent_messages
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import StreamingStdOutCallbackHandler

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해 주세요.")

DB_PATHS = ["./vectordb/test", "./vectordb/certification"]  # 벡터 DB 경로
MODEL_NAME = "BAAI/bge-large-en-v1.5"
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

# 초기화 (벡터스토어와 LLM 모델)
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

def load_and_merge_vectorstores(db_paths, embedding_model):
    vectorstore = None
    for db_path in db_paths:
        temp_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_serialization=True)
        if vectorstore is None:
            vectorstore = temp_store
        else:
            vectorstore.merge_from(temp_store)
    return vectorstore

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
            f"최근 질문: {infos}\n학생 질문: {user_message}\n답변:"
        """)
    return system_prompt_template

def get_rag_response(user_message: str, recent_messages: list) -> str:
    """
    RAG 기반 응답 생성 함수
    Args:
        user_message (str): 현재 사용자 질문
        recent_messages (list): 최근 질문 리스트
    Returns:
        str: RAG 기반 응답
    """
    try:
        # 최근 질문들을 결합하여 컨텍스트 생성
        combined_context = " ".join([msg["chatmessage"] for msg in recent_messages])
        system_prompt_template = apply_prompt_template()
        prompt_input = system_prompt_template.format(infos=combined_context, user_message=user_message)

        qa = ConversationalRetrievalChain.from_llm(
            llm=openai_model,
            retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True
        )
        response = qa.run(prompt_input)
        return response
    except Exception as e:
        raise ValueError(f"Error during RAG processing: {str(e)}")

def get_chatbot_response(user_message: str, username: str) -> str:
    """
    사용자 질문과 최근 질문을 기반으로 챗봇 응답 생성
    Args:
        user_message (str): 현재 사용자 질문
        username (str): 사용자 이름
    Returns:
        str: 챗봇 응답
    """
    # 최근 메시지 가져오기
    recent_messages = get_recent_messages(username)

    if recent_messages:
        print(f"최근 메시지: {recent_messages}")
        # 최근 질문을 포함하여 RAG 기반 응답 생성
        return get_rag_response(user_message, recent_messages)
    else:
        print("최근 메시지가 없습니다. 기본 방식으로 응답 생성.")
        # 최근 메시지가 없을 경우 기존 질문만으로 RAG 응답 생성
        try:
            response = get_rag_response(user_message, [])
            return response
        except Exception as e:
            print(f"기본 응답 생성 오류: {e}")
            return "죄송합니다, 응답 생성 중 문제가 발생했습니다."

# RAG 자원 초기화
initialize_rag_resources()
