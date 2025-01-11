import sqlite3
import pandas as pd
import os
import openai
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

class labeler:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, order, query):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": order,
                },
                {
                    "role": "user",
                    "content": f"""{query}
                                """,
                },
            ],
        )
        response = response.choices[0].message.content.strip()
        return response

def get_data():
    # DB 연결
    conn = sqlite3.connect('chat_data.db')
    cursor = conn.cursor()

    # 랜덤하게 1000개의 데이터 추출 쿼리
    query = "SELECT * FROM chat_messages ORDER BY RANDOM() LIMIT 1000"
    cursor.execute(query)

    # 데이터 가져오기
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns = ['author', 'message'])
    conn.close()
    return df

def labeling():
    messages = []
    gpt = labeler()
    labeled_datas = {'text': [], 'label' : []}
    df = get_data()
    for i, row in df[:50].iterrows():
        messages.extend(f"{i} : {row['message']}\n")
        labeled_datas['text'].append(row['message'])
        if (i+1) % 10 == 0:
            order = f"""주어지는 10개의 채팅이 욕설인지 아닌지 판별하세요.
                답변은 욕설(1), 비욕설(0)으로만 출력하세요.
                예시 : 1 0 0 1 1 0 0 1 0 1
                기준은 다음과 같습니다.
                <욕설>
                1. 비하성 채팅
                2. 조롱성 채팅 (88848, 쵸쵸쵸룰쵸, 888룰8 등 대회 성적 언급)
                3. 존나 못한다 등 비하의 의미로 사용된 비속어
                4. T1(티원), DRX(디알엑스), GEN(젠지), KT(케이티)와 같은 팀 이름을 언급하며 비하
                5. 선수 이름을 가리키며 비하
                <비욕설>
                1. 일상적인 대화
                2. 비속어가 포함되어 있어도 칭찬, 격려 등 의도가 명확한 경우
                3. 감탄사
                4. 화이팅과 같은 응원성 채팅
                """
            query = ''.join(messages)
            labels = gpt.generate(order, query)
            labels = labels.split(' ')
            labeled_datas['label'].extend(labels)
            messages = []
    labeled_df = pd.DataFrame(labeled_datas)
    return labeled_df