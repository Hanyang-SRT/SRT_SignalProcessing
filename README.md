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
일반적인 음성 인식 시스템은 단어 단위로 음성을 분석하지만, 한국어는 음절 단위로 억양과 성조의 변화가 있기 때문에 각 음절별로 pitch를 분석하였다. 이를 위해 먼저 whisper를 통해 제공받은 단어 단위 timestamp를 글자별로 분해한다. 이후 전체 단어 지속시간을 글자 수로 나누어 각 음절별로 pitch를 추출한 후, 음절을 구성하는 여러 음소들의 pitch값 중 중앙값을 해당 음절의 대표적인 pitch값으로 사용하였다. 이 과정에서 PYIN 알고리즘을 활용하여 사용자의 기본 주파수를 고려한 pitch를 추출하고자 하였다. PYIN 알고리즘은 성인 남성의 가장 낮은 음성부터 여성의 가장 높은 음성까지 모두 포괄하는 범위이기 때문에, 사람의 음성에서 기본 주파수를 찾는 데에 매우 효과적이다. 추출한 pitch값은 세미톤 단위로 정규화되어 각 화자별로 기준점을 설정하였다. 이 과정을 통해 사용자마다 다른 기본 주파수를 반영하여 절대적인 주파수값을 두 발화자의 상대적인 주파수 차이로 시각화하고자 하였다.

무성음 음소가 포함된 음절은 성대의 진동이 없이 나는 소리이기 때문에 pitch값을 절대적으로 추출하기 어렵다. 이 부분은 해당 음절의 앞뒤 음절을 활용하여 선형 보간을 통해 자연스럽게 이어지도록 하여 최종적으로 사용자와 원어민의 pitch를 비교하여 시각화할 때 보다 더 효율적으로 비교분석할 수 있도록 하였다. 
전체 점수는 100점에서 시작하여 세미톤 차이에 따라 단계별로 점수를 차감하였다. 3세미톤 이하의 차이는 자연스러운 개인차 범위로 보고 감점하지 않으며, 3-4세미톤은 2점, 4-6세미톤은 4점, 6세미톤 이상은 7점을 차감하는 단계별 채점 시스템을 적용하였다. 상대적인 pitch값의 유사도 차이에 따라 점수를 차감하여 직관적인 수치로 사용자의 현재 상태를 시각화하고자 하였다.

최종적으로 분석 결과를 시각화하여 음절을 가로축, 정규화된 pitch값을 세로축으로 하여 사용자와 원어민의 pitch 곡선을 함께 표시하여 사용자가 자신의 상태에 대해 효율적으로 파악할 수 있도록 하였다. 아래 사진은 시각화된 그래프의 예시이다.
<div align="center">
  <img src="https://github.com/user-attachments/assets/4d52469b-c6d7-4941-aa05-ffa6f9536096" width="70%">
  <br>
  [그림 2] pith 분석 시각화 결과
</div>
시각화된 분석 결과는 base64로 인코딩되어 웹에서 바로 표시할 수 있도록 처리하였다.

### Intensity Analysis
### Duration Anaylsis

## Role Contribution 
| Team Member | Role |
|-------------|------|
| **Areum Kim** | • Intensity & Duration Feature Analysis<br>• Similarity Evaluation<br>• Client-to-Server API Request Handling |
| **Na-young Lee** | • Pitch Feature Analysis<br>• Design JSON Data Pipeline<br>• Functional and Performance Design |

본 프로젝트는 한양대학교 에리카 캠퍼스 인공지능학과 음성인식의 IC-PBL 프로젝트이다.
* Areum Kim : dkfma0817@hanyang.ac.kr
* Na-young Lee : lwg2326@hanyang.ac.kr
