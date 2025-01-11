from chat_crawler import Extract
from preprocessing import Transform
from load_database import Load
from argparse import Namespace
from model import _1DCNN_MODEL
import pandas as pd
from sqlalchemy import create_engine
if __name__ == '__main__':
    Extract()
    Transform()
    Load()

    args = Namespace(
    EPOCH=5,
    batch_size=32,
    emb_model_name="klue/bert-base"
    )
    model = _1DCNN_MODEL(args)
    engine = create_engine('sqlite:///chat_data.db')
    print('추가된 데이터를 기반으로 학습합니다.')
    existing_routes_query = """
    SELECT * FROM labeled_chat_messages
    """
    train_data = pd.read_sql(existing_routes_query, con=engine)
    model.train(train_data)
