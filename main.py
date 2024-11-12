import os
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "sk-svcacct-IQFqO84FdCKXC8s9ZJikTypIAVnRknNseMZVdhjxVwKw1b7v-7hNWsfRHw1YYiT3BlbkFJ6IhVPGelfVBzlliX6tZlasIfcRQJWdJpdZA2VUpuSAklBDBcGnT_BrTaJmFGEA"

# ChatOpenAI 모델 초기화
chat_openai = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0
)

# 챗봇과 상호작용할 함수 정의
def chatbot_interaction():
    print("챗봇과 대화를 시작합니다. 종료하려면 'q'를 입력하세요.")
    while True:
        # 사용자 질문 입력
        user_input = input("나: ")
        if user_input.lower() == 'q':
            print("대화를 종료합니다.")
            break

        # 질문에 대한 답변 생성
        response = chat_openai.predict(user_input)
        print("챗봇:", response)

# 챗봇 상호작용 함수 실행
chatbot_interaction()

