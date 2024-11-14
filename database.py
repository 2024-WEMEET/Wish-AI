import os
import mysql.connector
from mysql.connector import Error 
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def create_connection():
    try:
        # MySQL 연결 생성 (백엔드에서 제공한 키 정보로 설정)
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),       # MySQL 서버 호스트
            user=os.getenv("MYSQL_USER"),       # 사용자 이름
            password=os.getenv("MYSQL_PW"), # 비밀번호
            database=os.getenv("MYSQL_DATABASE"),  # 데이터베이스 이름
            port=int(os.getenv("MYSQL_PORT","3306")) 
        )
        if connection.is_connected():
            print("MySQL 데이터베이스에 성공적으로 연결되었습니다.")
            return connection
    except Error as e:
        print(f"Error: {e}")
        return None

def close_connection(connection):
    if connection and connection.is_connected():
        connection.close()
        print("MySQL 연결이 닫혔습니다.")
