import pytchat
import csv
import pandas as pd
from datetime import datetime
import requests
import pafy
import requests
import os
from googleapiclient.discovery import build
from typing import List
from dotenv import load_dotenv
load_dotenv()
def get_channel_id(channel_name):

    DEVELOPER_KEY = os.getenv('YOUTUBE_API_KEY')
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
    search_response = youtube.search().list(
        q = channel_name,
        order = "relevance",
        part = "snippet",
        maxResults = 50
        ).execute()

    channel_id =search_response['items'][0]['id']['channelId']

    return channel_id


def get_playlist_ids(channel_id):
    # 환경 변수에서 API 키 불러오기
    api_key = os.getenv("YOUTUBE_API_KEY")

    # 채널 ID를 사용하여 해당 채널의 플레이리스트 목록을 가져오기 위한 URL
    url = "https://www.googleapis.com/youtube/v3/playlists"
    params = {
        "part": "snippet",  # 플레이리스트 제목과 기타 정보를 포함한 부분
        "channelId": channel_id,
        "maxResults": 100,  # 한번에 가져올 플레이리스트 수
        "key": api_key  # API 키
    }

    # API 요청 보내기
    response = requests.get(url, params=params)

    # 응답이 성공적일 경우
    if response.status_code == 200:
        playlists = response.json().get("items", [])
        playlist_info = []
        
        # 각 플레이리스트의 ID와 제목을 가져오기
        for playlist in playlists:
            playlist_id = playlist["id"]
            playlist_title = playlist["snippet"]["title"]
            playlist_info.append({"id": playlist_id, "title": playlist_title})

        return playlist_info
    else:
        print(f"API 요청 실패: {response.status_code} - {response.text}")
        return []

def get_video_id(playlist_id):
    base_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    params = {
    "part": "contentDetails",
    "playlistId": playlist_id,
    "key": os.getenv('YOUTUBE_API_KEY'),
    "maxResults": 10}

    response = requests.get(base_url, params = params)
    if response.status_code == 200:
        data = response.json()
        video_l = [item['contentDetails']['videoId'] for item in data['items']]
    
    else:
        print("Failed to fetch playlist video IDs:", response.text)
    return video_l

def crawling(video_l:List):
    os.makedirs('chattings', exist_ok=True)
    already_crawled = os.listdir('chattings')
    for video_id in video_l:
        try:
            url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key=AIzaSyCaOZGugJAXTNoTT46O_rfEZRBouKkvUUs"
            response = requests.get(url)
            data = response.json()
           
            date = data['items'][0]['snippet']['publishedAt'][:10]
            title = data['items'][0]['snippet']['title']
            id = data['items'][0]['id']
            file_name = f'chattings/{date}_{title}.csv'
            if file_name in already_crawled:
                print(title, '는 이미 크롤링한 영상입니다.')
                continue
            chat = pytchat.create(video_id = id)
            chat_l = {'name' : [], 'message' : []}
            print(title,'의 채팅 크롤링 진행중')
            idx = 0
            while chat.is_alive():
                try:  
                    data = chat.get()
                    items = data.items
                    for c in items:
                        name = c.author.name if hasattr(c.author, 'name') else 'Unknown Author'
                        m = c.message
                        chat_l['name'].append(name)
                        chat_l['message'].append(m)
                except KeyboardInterrupt:
                    chat.termincate()
                    break
            chat_l = pd.DataFrame(chat_l)
            if len(chat_l) == 0:
                print(title, '에서 오류 발생')
                continue
            else:
                chat_l.to_csv(file_name, index = False)
                print('완료')
        except IndexError as e:
            print(video_id,'에서의 오류',e)
    print('모든 작업 완료.')
    
def Extract():
    channel_id = get_channel_id('LCK')
    playlist_ids = get_playlist_ids(channel_id)
    df = pd.DataFrame(playlist_ids)
    FULL_VOD = df[df['title'].str.contains('FULL')]['id'].values
    for playlist_id in FULL_VOD:
        video_l = get_video_id(playlist_id)
        crawling(video_l)

if __name__ == '__main__':
    Extract()