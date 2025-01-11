import re
import itertools
from typing import List
from glob import glob
import pandas as pd


def erase_bot(df):
    idxs = df[df['message'].str.startswith('@') == True].index
    df = df.drop(idxs)
    bot_cnt = len(idxs)
    print('message 개수 : ', len(df['message']), '처리한 봇 수 : ', bot_cnt, '\n')
    return df
# erase_bot 함수 사용 예시
# data = erase_bot(data)

def null_finder(df): # 공백만 있는 메세지 제거
    print('None 제거 전:', len(df['message']))
    df['message'] = df['message'].dropna()
    print('None 제거 후:', len(df['message']))
    return df

def len_control(df):
    print('길이가 너무 작거나 큰 메세지를 제거합니다.')
    print('제거 전:', len(df))
    df = df[df['message'].apply(lambda x: 2 <= len(str(x)) < 50)]
    print('제거 후:', len(df))
    return df

def remain_kor_eng_num(df):
    # 한글, 영어, 숫자만 남겨서 새로운 메시지로 갱신
    df['message'] = df['message'].apply(lambda x: ''.join(re.findall(r'[ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z0-9\s]', str(x))))
    return df

def erase_emoji(df):
    df['message'] = df['message'].apply(lambda x: re.sub(r':[\w-]+:', '', str(x)))
    return df


# 중복된 문자(두 번 이상 반복된 문자)를 하나로 줄이기
def long2short(df):
    result = []
    
    for ele in df['message']:
        # 반복되는 문자 그룹을 처리
        while True:
            # 반복되는 문자를 찾아서 그룹화
            candidates = set(re.findall(r'(\w)\1+', ele))  # 예: 'aa', 'bb' 등
            repeats = itertools.chain(*[re.findall(r"({0}{0}+)".format(c), ele) for c in candidates])

            # 반복된 문자를 첫 번째 문자로 줄임
            updated = False
            for org in repeats:
                if len(org) >= 2:  # 2개 이상의 반복만 처리
                    ele = ele.replace(org, org[0])  # 반복되는 문자 중 첫 번째 문자로 교체
                    updated = True
            
            # 더 이상 변화가 없으면 종료
            if not updated:
                break
                
        result.append(ele)  # 결과 리스트에 추가
    
    df['message'] = result  # 처리된 메시지로 업데이트
    return df

def preprocessing(df):
    df = erase_bot(df)
    df = erase_emoji(df)
    df = remain_kor_eng_num(df)
    df = long2short(df)
    df = null_finder(df)
    df = len_control(df)
    return df

def Transform():
    routes = glob('chattings/*.csv')
    for r in routes:
        name = r.split('/')[-1] 
        print(name, '파일 내의 메세지를 전처리합니다.')
        df = pd.read_csv(r)
        df = preprocessing(df)
        df.to_csv(r, index = False)
    
    print('chattings 폴더 내 csv 파일 전처리 완료')

if __name__ == '__main__':
    Transform()