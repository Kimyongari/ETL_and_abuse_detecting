# **LCK Chatting ETL Pipeline**

> LCK 채널의 채팅 데이터를 **Extract, Transform, Load**하는 파이프라인입니다.  
이후 수집된 데이터를 기반으로 **1D-CNN 모델**을 학습시켜 **욕설 탐지 모델**을 개발합니다.  
**(모델 파라미터 수: 약 2500만 개, 모델 용량: 약 100MB)**

---

## **Pipeline Steps**

### **1. Extract**
1. **LCK 채널의 최신 플레이리스트** 100개의 재생목록 ID를 가져옵니다.  
2. **채팅 다시보기를 지원하는 FULL VOD 재생목록**의 Video ID를 수집합니다.  
3. 각 Video에서 채팅 데이터를 수집하고, **`chattings/동영상_제목.csv`** 형태로 저장합니다.

---

### **2. Transform**
1. 수집된 채팅 데이터를 **전처리**합니다.  
   전처리 과정은 다음 순서로 이루어집니다:  
   - **봇 채팅 제거**
   - **이모지 제거**
   - **한국어, 영어만 남기기**
   - **중복 제거**
   - **null 값 제거**
   - **너무 길거나 짧은 문장 제거**

---

### **3. Load**
1. 전처리된 채팅 데이터를 **`chat_data.db`**에 저장합니다.  
2. 라벨링되지 않은 데이터를 추론하기 위해 **이전 모델**과 **GPT 4o mini**를 활용합니다.  
   - 약 50개의 데이터를 샘플링하여 **두 모델의 추론 결과가 일치하는 데이터**를 `labeled_chat_data`에 추가합니다.  
   - 샘플링 수를 늘리고 싶다면, **`labeler.py`의 52번째 줄**을 수정하세요.  
     ```python
     df[:50].iterrows()  # 이 값을 변경하여 샘플링 개수 조정
     ```

---

## **Database Structure: `chat_data.db`**

### **1. Table: `games`**
| Column Name | Data Type | Description              |
|-------------|-----------|--------------------------|
| `games`     | TEXT      | 수집된 게임의 제목        |

---

### **2. Table: `labeled_chat_data`**
| Column Name | Data Type | Description                     |
|-------------|-----------|---------------------------------|
| `text`      | TEXT      | 채팅 내용                        |
| `label`     | INTEGER   | 라벨 (1: 욕설, 0: 정상)          |

---

### **3. Table: `chat_messages`**
| Column Name | Data Type | Description               |
|-------------|-----------|---------------------------|
| `name`      | TEXT      | 채팅 작성자               |
| `message`   | TEXT      | 채팅 내용                 |

---

## **Model Training**
1. **라벨링된 데이터**를 바탕으로 **1D-CNN 모델**을 학습합니다.  
2. 모델 사양:
   - **파라미터 수**: 약 2500만 개  
   - **모델 크기**: 약 100MB  
3. **학습 환경**:
   - 맥북 M2 Air (RAM 8GB) 기준 **학습 시간: 1분 이내**  
4. **훈련 데이터 갱신** 시 모델을 재학습합니다.  
5. **평가 메트릭**: 
   - Accuracy (ACC)  
   - F1 Score  
   - ROC-AUC Score  
6. **최고 성능 모델**의 메트릭은 `results/best_model_metrics.json`에 저장되며, 갱신 시 모델도 함께 저장됩니다.

---

## **Seed Train Data**
1. 모델 훈련을 위해 약 **10,000개**의 채팅 데이터를 직접 라벨링하였습니다.  
2. 라벨 분포를 맞추기 위해 **8,000개**를 필터링하여 **2,000개의 데이터**를 최종 선정했습니다.  
3. **욕설 기준**:
   - 팀, 선수에 대한 비하
   - 욕설이 포함되지 않더라도 당사자가 듣기에 **기분 나쁠 만한 채팅**
   - 선수의 기량, 대회 성적 등을 들먹이는 채팅
   - **예외**: 비속어가 포함되어도 칭찬성 채팅은 욕설로 라벨링하지 않음  
     - 예: "와 ㅈㄴ 잘한다"

---

## **Model Performance**
- **Seed Train Data**(2,000개 기준)에서의 모델 성능:  
  - **ACC**: 0.9871  
  - **F1 Score**: 0.9857  
  - **ROC-AUC Score**: 0.9867  
- 학습은 **5 epochs** 기준으로 진행되었습니다.

---

### **Contact**
ETL 파이프라인과 모델 관련 문의 사항은 담당자에게 문의하세요.  