import pandas as pd
from sqlalchemy import create_engine
from glob import glob
import sqlite3
from labeler import labeling
from model import _1DCNN_MODEL
from argparse import Namespace
import os

def search_new():
    # CSV 파일 경로 읽기
    routes = glob('chattings/*.csv')

    # SQLite 데이터베이스 연결
    engine = create_engine('sqlite:///chat_data.db')

    # games 테이블이 없으면 생성
    existing_routes_query = """
    SELECT name FROM sqlite_master;
    """
    games_exists = pd.read_sql(existing_routes_query, con=engine)

    if 'games' not in games_exists['name'].values:
        # games 테이블이 없을 경우 새로 생성
        pd.DataFrame({'games': []}).to_sql('games', con=engine, if_exists='replace', index=False)

    if 'labeled_chat_messages' not in games_exists['name'].values:
        pd.read_csv('labeled_data_balanced.csv').to_sql('labeled_chat_messages', con = engine, if_exists = 'append', index = False)

    # games 테이블에서 기존 경로 가져오기
    existing_games_query = "SELECT games FROM games;"
    existing_routes = pd.read_sql(existing_games_query, con=engine)

    # games 테이블에 없는 경로 필터링
    existing_routes_list = existing_routes['games'].tolist()
    new_routes = [r for r in routes if r not in existing_routes_list]

    # 새 경로를 games 테이블에 추가
    if new_routes:
        new_games_df = pd.DataFrame({'games': new_routes})
        new_games_df.to_sql('games', con=engine, if_exists='append', index=False)
        print("games 테이블에 없는 새로운 경로가 추가되었습니다:")
        print(new_routes)
        return new_routes
    else:
        print('추가된 데이터가 없습니다.')
        return None
    
def save_datas(routes):
    engine = create_engine('sqlite:///chat_data.db')
    if routes:
        for r in routes:
            df = pd.read_csv(r)
            df.to_sql('chat_messages', con = engine, if_exists = 'append', index = False)
        print("새로운 채팅데이터들이 데이터베이스에 저장되었습니다.")
    else:
        print('새롭게 갱신된 채팅 데이터가 없습니다.')
    print('정답 데이터를 추가합니다.')

    labeled_df = labeling()
    labeled_df['label'] = labeled_df['label'].astype(int)

    args = Namespace(
    EPOCH=5,
    batch_size=32,
    emb_model_name="klue/bert-base"
)
    model = _1DCNN_MODEL(args)
    if os.path.isdir('results'):
        model.load_best_model()
    else:
        existing_routes_query = """
            SELECT * FROM labeled_chat_messages
            """
        train_data = pd.read_sql(existing_routes_query, con=engine)
        print('학습된 모델이 없어 SEED 데이터로 새 모델을 학습합니다.')
        model.train(train_data)
    logits = model.predict(labeled_df['text'].to_list())
    predictions = (logits > 0.5).int().view(-1).tolist()
    filtered_df = labeled_df[labeled_df['label'] == predictions]
    print('GPT의 라벨링과 모델의 라벨링이 일치하는 데이터만 Labeled Data로 추가합니다.')
    print('추가된 데이터 수:', len(filtered_df))
    filtered_df.to_sql('labeled_chat_messages', con = engine, if_exists = 'append', index = False)
    query = "SELECT count(text) FROM labeled_chat_messages;"
    print('현재 라벨링 데이터:', pd.read_sql(query, con=engine).loc[0].item())

def Load():
    routes = search_new()
    save_datas(routes)

if __name__ == '__main__':
    Load()