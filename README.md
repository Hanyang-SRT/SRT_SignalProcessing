# SRT_SignalProcessing
본 프로젝트는 Prosody-Based Korean Speech Analysis and Feedback System for Foreign Learners를 목표로 하며, 이 레퍼지토리는 그 중 음성 신호 분석에 대한 내용이 담겨있다.

## Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [Libraries and Tools](#libraries-and-tools)
- [Features](#features)
- [Role Contribution](#role-contribution)
  
## Overview
본 프로젝트는 외국인 한국어 학습자를 위한 Korean Transcriber를 주제로 하는 프로젝트로, 그 중 한국어의 운율을 한국 드라마와 같은 한류 콘텐츠 섀도잉을 통해 학습하는 것을 목표로 한다. 한국어에서 운율이란 성조, 억양, 강세, 리듬 등을 포괄하는 용어로 의미상의 차이를 가져오는 소리의 특징을 말한다. 한국어의 운율 정보는 청자들로 하여금 단어와 구를 식별하는 데에 영향을 끼치며, 실시간 언어 처리를 효율적으로 만들어 말소리를 보다 잘 이해하는 데에 도움을 줄 수 있다. 

사용자 타겟층은 한국어 발음에는 무리를 느끼지 않는 외국인 학습자이며, 학습을 위한 한류 콘텐츠는 모두 표준어를 기반으로 한다. 사용자는 콘텐츠 속 원어민 인물들의 발화와 자신의 발화를 비교한 결과값을 제공받을 수 있으며, 사용자 발화의 운율을 분석하는 과정은 아래에 기재되어 있다.

## Project Structure
```
SRT_SignalProcessing
  ┣ 📂 media
  ┃ ┗ pitch.png  // pitch 분석 시각화 파일
  ┣ app.py
  ┣ pitch_processing.py  
  ┣ intensity_processing.py
  ┣ duration_processing.py
  ┣ integrated_processing.py  // 세 가지 운율 추출 분석 통합
  ┣ movieclip.ipynb  // 영상 클립 분할
  ┣ 📄 README.md
  ┗ 📄 .gitignore
```

## Development Environment
* Python version : 3.10.5
* CPU : Apple M2

## Libraries and Tools
* librosa version : 0.11.0
* soundfile version : 0.13.1
* numpy version : 2.1.3
* scipy version : 1.15.2
* torch version : 2.6.0
* fastdtw version 0.3.4
* matplotlib version : 3.10.1
* transformers version : 4.51.3
* fastapi version : 0.115.12

해당 라이브러리는 다음과 같은 명령어를 통해 설치할 수 있다.
```bash
pip install librosa==0.11.0 soundfile==0.13.1 numpy==2.1.3 scipy==1.15.2 torch==2.6.0 transformers==4.51.3 fastapi==0.115.12 fastdtw==0.3.4 matplotlib==3.10.1 uvicorn python-multipart yt-dlp
```

## Features

모든 운율을 분석하는 데에 있어서 음성인식 모델로는 openai의 whisper를 사용하였으며, 사용자의 음성은 spectral subtraction을 통해 노이즈 전처리 과정을 거친 후 운율을 분석한다. 다음은 운율을 분석하는 과정을 나타낸 플로우 차트이다. 
<div align="center">
  <img src="https://github.com/user-attachments/assets/f86b0ddb-0f60-470b-9166-10798f0de3f2" width="70%">
  <br>
  [그림 1] 운율 분석 처리 과정
</div>

### Pitch Analysis

- **음절 단위 Pitch 분석**: 한국어 억양 특성을 반영해, 단어를 음절로 분해 후 각 음절별 pitch 계산
- **기법**: PYIN 알고리즘으로 pitch 추출 → 중간값을 대표 pitch로 사용
- **보정**: 무성음 음절은 선형 보간(interpolation)으로 자연스럽게 연결
- **채점 방식**:
  - 기준: 100점 만점
  - 3세미톤 이내: 감점 없음
  - 3–4 세미톤: –2점, 4–6: –4점, 6↑: –7점 감점
- **결과 시각화**: 음절별 pitch 곡선(사용자 vs 원어민)을 그래프로 표시

<div align="center">
  <img src="https://github.com/user-attachments/assets/d4ad3781-8fdb-4231-ad65-5e6171641572" width="70%">
  <br>
  [그림 2] Pitch 분석 시각화 예시
</div>
<br>
</div>
시각화된 분석 결과는 base64로 인코딩되어 웹에서 바로 표시할 수 있도록 처리하였다.

### Intensity Analysis

- **단어 단위 강세 분석**: Whisper의 단어별 timestamp로 각 구간을 분할하고 RMS 에너지 계산
- **기법**: Librosa를 사용하여 각 단어 구간의 평균 에너지를 추출
- **정규화**: 문장 내 상대적인 강조 정도를 보기 위해 z-score 정규화 적용
- **채점 기준**:
  - 각 단어마다 z-score 차이 `diff`에 대해 `1 - |diff| / 3` 방식으로 점수를 계산
  -  전체 단어 평균 점수를 100점 만점으로 환산



### Duration Analysis

- **단어 발화 길이 분석**: Whisper의 timestamp로 각 단어의 발화 지속시간 계산
- **기법**: 사용자와 원어민의 각 단어 duration을 계산하여 DTW(Dynamic Time Warping)로 정렬 후 비교
- **정규화**: 전체 문장 길이를 1로 두고 상대적인 비율로 각 단어의 duration 환산
- **보정**: 문장의 처음과 마지막 단어에서 무성음으로 인한 공백 구간은 제거하여 실제 발화 구간만 반영
- **채점 기준**:
  - fastDTW 결과로 정렬된 사용자-원어민 duration쌍의 차이를 기반으로 MAE(Mean Absolute Error) 계산
  - 점수 = `max(0, 100 × (1 - MAE))`
  - MAE가 클수록 감점

#### Final Result
최종적으로 분석된 사용자의 음성은 다음과 같이 json 형식으로 반환되며, 각 분석 결과별로 사용자 음성, 원어민 음성, 평가 결과, 피드백 메세지를 포함하여 웹 서버로 전달된다. 이 데이터는 프론트엔드에서 실시간 시각화 및 학습자 피드백 제공에 활용된다.

```bash
{
  {
  "pitch": {
    "user": [ -2.25, 1.2, -0.2, 0, -0.1, 0.6, -1.2, 2.9, 2, 2, -2.3],
    "native": [ -2.15, 1.3 , -0.1, 0.1, 0, -0.15, -1.1, 3, 2.1, 2.1,-2.2],
    "score": 96
  },
  "image": "iVBORw0KGgoAAAANSUhE~~",
  {
  'stress': {
    'user': [-0.85, -0.7, 0.82, 1.56, -0.82], 
    'native': [0.69, 0.1, -1.75, 1.19, -0.22], 
    'score': 63, 
    'highlight': [True, False, True, False, False], 
    'feedback': ["' 얼른' 단어가 약하게 발음되었습니다.", "' 지금' 단어에 불필요한 강조가 있습니다."]
  }, 
  'duration': {
    'user': [0.21, 0.15, 0.45, 0.56, 0.42], 
    'native': [0.25, 0.28, 0.64, 0.5, 0.17],
    'score': 93, 
    'highlight': [False, False, False, False, True],
    'feedback': ["' 돼' 단어를 상대적으로 길게 발음했습니다."]
    },
  "user_text": " 얼른 가자 지금 출발해야 돼"
}
```

## Role Contribution 
| Team Member | Role |
|-------------|------|
| **Areum Kim** | • Intensity & Duration Feature Analysis<br>• Similarity Evaluation<br>• Client-to-Server API Request Handling |
| **Na-young Lee** | • Pitch Feature Analysis<br>• Design JSON Data Pipeline<br>• Functional and Performance Design |

본 프로젝트는 한양대학교 에리카 캠퍼스 인공지능학과 음성인식의 IC-PBL 프로젝트이다.
* Areum Kim : dkfma0817@hanyang.ac.kr
* Na-young Lee : lwg2326@hanyang.ac.kr
