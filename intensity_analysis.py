from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from g2pk import G2p
import logging
import traceback
import torch
import os
import io
import base64

# MPS 문제 방지
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.backends.mps.is_available = lambda: False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 단어별 intensity 계산
def get_intensity_per_chunk(audio_file, chunks):
    # try:
    logger.info("Intensity 계산 시작")
    y, sr = librosa.load(audio_file, sr=None)
    intensities = []
    words = []
    
    for chunk in chunks:
            
        start_time, end_time = chunk['timestamp']
        if start_time is None or end_time is None:
            continue
            
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        rms = np.sqrt(np.mean(segment**2)) if len(segment) > 0 else 0.0
        intensities.append(rms)
        words.append(chunk['text'])
        
    logger.info(f"Intensity 계산 완료: {len(intensities)}개 값")
    return words, intensities

# 강세 비교 분석
def compare_intensity(words_ref, intensities_ref, words_usr, intensities_usr):
    logger.info("강세 분석 시작")
    if not intensities_ref or not intensities_usr:
        return [], [], 0, [], ["음성 데이터가 부족합니다."]
        
    ref_z = zscore(intensities_ref) if len(intensities_ref) > 1 else np.zeros_like(intensities_ref)
    usr_z = zscore(intensities_usr) if len(intensities_usr) > 1 else np.zeros_like(intensities_usr)

    feedbacks = []
    highlights = []
    total_score = 0
    count = min(len(ref_z), len(usr_z))

    for i in range(count):
        diff = usr_z[i] - ref_z[i]
        score = max(0.0, 1.0 - abs(diff) / 2.0)
        total_score += score

        if diff > 1.0:
            feedbacks.append(f"'{words_usr[i]}' 단어에 불필요한 강조가 있습니다.")
            highlights.append(True)
        elif diff < -1.0:
            feedbacks.append(f"'{words_usr[i]}' 단어가 약하게 발음되었습니다.")
            highlights.append(True)
        else:
            highlights.append(False)

    avg_score = int(round((total_score / count) * 100)) if count else 0

    if avg_score >= 85 and not feedbacks:
        feedbacks.append("전반적으로 강세를 매우 잘 따라했습니다!")
    elif avg_score >= 75 and not feedbacks:
        feedbacks.append("강세 전달이 자연스럽고 좋습니다!")

    logger.info(f"강세 분석 완료: 점수 {avg_score}")
    return usr_z.tolist(), ref_z.tolist(), avg_score, highlights, feedbacks

def run_intensity_analysis(user_denoised_path, ref_denoised_path, user_chunks, ref_chunks):
    """ intensity 분석 함수 """
    try:
        # 입력 유효성 검사
        if not user_chunks or not ref_chunks:
            logger.error("청크 데이터가 비어있음")
            return {"error": "청크 데이터가 비어있음"}
        
        if not os.path.exists(user_denoised_path) or not os.path.exists(ref_denoised_path):
            logger.error("오디오 파일이 존재하지 않음")
            return {"error": "오디오 파일이 존재하지 않음"}

        # 청크 정렬
        user_chunks_sorted = sorted(user_chunks, key=lambda x: x['timestamp'][0] if x.get('timestamp') else float('inf'))
        ref_chunks_sorted = sorted(ref_chunks, key=lambda x: x['timestamp'][0] if x.get('timestamp') else float('inf'))
        
        logger.info("Intensity 분석 시작")
        words_usr, intensities_usr = get_intensity_per_chunk(user_denoised_path, user_chunks_sorted)
        words_ref, intensities_ref = get_intensity_per_chunk(ref_denoised_path, ref_chunks_sorted)

        usr_z, ref_z, stress_score, stress_highlights, stress_feedbacks = compare_intensity(
            words_ref, intensities_ref, words_usr, intensities_usr
        )

        def safe_convert(value, default=None):
            try:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return default
                return float(value) if isinstance(value, (int, float, np.number)) else value
            except:
                return default

        result = {
            "intensity": {
                "user": [round(safe_convert(z, 0), 2) for z in usr_z],
                "native": [round(safe_convert(z, 0), 2) for z in ref_z],
                "score": int(safe_convert(stress_score, 0)),
                "highlight": [bool(h) for h in stress_highlights],
                "feedback": list(stress_feedbacks)
            }
        }
        
        logger.info(f"Intensity 분석 완료 - 점수: {stress_score}")
        return result

    except Exception as e:
        logger.error(f"Intensity 분석 실패: {e}")
        traceback.print_exc()
        return {"error": f"Intensity 분석 실패: {str(e)}"}