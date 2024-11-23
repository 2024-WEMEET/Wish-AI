import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

def create_connection():
    """
    MySQL 데이터베이스 연결 생성
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),       # MySQL 서버 호스트
            user=os.getenv("MYSQL_USER"),       # 사용자 이름
            password=os.getenv("MYSQL_PW"),     # 비밀번호
            database=os.getenv("MYSQL_DATABASE"), # 사용할 데이터베이스
            port=int(os.getenv("MYSQL_PORT", "3306"))  # 포트 번호 (기본값: 3306)
        )
        if connection.is_connected():
            print("MySQL 연결 성공")
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def close_connection(connection):
    """
    MySQL 연결 종료
    """
    if connection and connection.is_connected():
        connection.close()
        print("MySQL 연결 닫힘")

def get_recent_messages(username):
    """
    특정 사용자의 최근 3개 메시지를 MySQL에서 가져오는 함수
    Args:
        username (str): 사용자 이름
    Returns:
        list: 최근 메시지 3개를 담은 딕셔너리 리스트
    """
    query = """
    SELECT chatmessage, timestamp
    FROM messages
    WHERE username = %s
    ORDER BY timestamp DESC
    LIMIT 3
    """
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, (username,))
            results = cursor.fetchall()  # [{"chatmessage": ..., "timestamp": ...}, ...]
            return results
        except Error as e:
            print(f"Error fetching recent messages: {e}")
            return []
        finally:
            cursor.close()
            close_connection(connection)
