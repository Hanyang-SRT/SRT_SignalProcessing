import re
import numpy as np
import librosa
import soundfile as sf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import logging
import traceback
import torch
import os
import io
import base64

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_korean_syllable(char):
    """한글 음절인지 확인하는 함수"""
    return '가' <= char <= '힣'

def extract_syllables_from_chunks(chunks):
    """Whisper 청크에서 음절 단위로 타임스탬프를 추출하는 함수"""
    try:
        syllables = []
        for chunk in chunks:
            word = chunk["text"].strip()
            start, end = chunk["timestamp"]
            duration = end - start
            
            if duration <= 0:
                continue
            
            # 단어를 글자 단위로 분리
            word_syllables = [c for c in word if c.strip()]
            
            if not word_syllables:
                continue
            
            # 각 글자에 균등하게 시간 할당
            # 이는 완벽하지 않지만 실용적인 근사치를 제공합니다
            syllable_duration = duration / len(word_syllables)
            
            for i, syllable in enumerate(word_syllables):
                syllable_start = start + i * syllable_duration
                syllable_end = syllable_start + syllable_duration
                
                syllables.append({
                    "text": syllable,
                    "timestamp": [syllable_start, syllable_end],
                    "word_origin": word  # 디버깅용 원본 단어 정보
                })
        
        logger.info(f"음절 추출 완료: {len(syllables)}개 음절")
        return syllables
        
    except Exception as e:
        logger.error(f"음절 추출 실패: {e}")
        return []

def extract_pitch_from_syllables(audio_path, syllables):
    """음절별로 대표 피치 값을 추출하는 함수"""
    try:
        logger.info("음절별 Pitch 추출 시작")
        
        # 오디오 파일 로드
        y, sr = librosa.load(audio_path, sr=None)
        
        # PYIN 알고리즘으로 피치 추출
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),  # 약 65Hz (남성 저음)
            fmax=librosa.note_to_hz('C7'),  # 약 2093Hz (여성 고음)
            sr=sr
        )
        
        # 피치 데이터의 시간 축 생성
        times = librosa.times_like(f0, sr=sr)
        
        pitches = []
        labels = []
        
        for syllable in syllables:
            if not syllable.get('timestamp'):
                pitches.append(np.nan)
                labels.append(syllable.get('text', ''))
                continue
            
            start_time, end_time = syllable['timestamp']
            
            # 해당 시간 구간의 인덱스 찾기
            start_idx = np.argmin(np.abs(times - start_time))
            end_idx = np.argmin(np.abs(times - end_time))
            
            # 해당 구간의 피치 데이터 추출
            pitch_segment = f0[start_idx:end_idx+1]
            
            # 유효한 피치 값만 필터링 (NaN 제거)
            valid_pitches = pitch_segment[~np.isnan(pitch_segment)]
            
            # 대표 피치 계산: 중간값 사용
            if len(valid_pitches) > 0:
                representative_pitch = np.median(valid_pitches)
            else:
                # 유효한 피치가 없는 경우 (무성음, 침묵 등)
                representative_pitch = np.nan
            
            pitches.append(representative_pitch)
            labels.append(syllable['text'])
        
        logger.info(f"Pitch 추출 완료: {len(pitches)}개 음절")
        return pitches, labels
        
    except Exception as e:
        logger.error(f"Pitch 추출 실패: {e}")
        return [], []

def normalize_pitch_for_comparison(user_pitches, ref_pitches):
    """두 화자 간의 피치 차이를 정규화하는 함수"""
    try:
        # 유효한 피치 값만 추출 (NaN과 0 이하 값 제외)
        user_valid = np.array([p for p in user_pitches if not np.isnan(p) and p > 0])
        ref_valid = np.array([p for p in ref_pitches if not np.isnan(p) and p > 0])
        
        if len(user_valid) == 0 or len(ref_valid) == 0:
            logger.warning("정규화할 유효한 피치 데이터가 부족함")
            return user_pitches, ref_pitches
        
        # 각자의 기준 주파수 계산
        user_baseline = np.median(user_valid)
        ref_baseline = np.median(ref_valid)
        
        logger.info(f"기준 주파수 - User: {user_baseline:.1f}Hz, Reference: {ref_baseline:.1f}Hz")
        
        # 세미톤 단위로 정규화
        user_normalized = []
        ref_normalized = []
        
        for pitch in user_pitches:
            if np.isnan(pitch) or pitch <= 0:
                user_normalized.append(np.nan)
            else:
                semitones = 12 * np.log2(pitch / user_baseline)
                user_normalized.append(semitones)
        
        for pitch in ref_pitches:
            if np.isnan(pitch) or pitch <= 0:
                ref_normalized.append(np.nan)
            else:
                semitones = 12 * np.log2(pitch / ref_baseline)
                ref_normalized.append(semitones)
        
        return user_normalized, ref_normalized
        
    except Exception as e:
        logger.error(f"피치 정규화 실패: {e}")
        return user_pitches, ref_pitches

def align_syllables_by_sequence(user_syllables, user_pitches, ref_syllables, ref_pitches):
    """음절 시퀀스를 순서대로 정렬하는 함수"""
    try:
        aligned_user_pitches = []
        aligned_ref_pitches = []
        aligned_labels = []
        
        # 더 짧은 시퀀스에 맞춰 정렬
        min_length = min(len(user_syllables), len(ref_syllables))
        
        for i in range(min_length):
            user_syllable = user_syllables[i]
            ref_syllable = ref_syllables[i]
            user_pitch = user_pitches[i] if i < len(user_pitches) else np.nan
            ref_pitch = ref_pitches[i] if i < len(ref_pitches) else np.nan
            
            # 같은 위치의 음절끼리 매칭
            aligned_user_pitches.append(user_pitch)
            aligned_ref_pitches.append(ref_pitch)
            aligned_labels.append(ref_syllable)
            
            # 음절이 다른 경우 로그 출력
            if user_syllable != ref_syllable:
                logger.debug(f"음절 불일치 - 위치 {i}: '{user_syllable}' vs '{ref_syllable}'")
        
        logger.info(f"음절 정렬 완료: {len(aligned_labels)}개 음절 매칭")
        return aligned_user_pitches, aligned_ref_pitches, aligned_labels
        
    except Exception as e:
        logger.error(f"음절 정렬 실패: {e}")
        return [], [], []

def calculate_pitch_score(user_pitch, ref_pitch, threshold=3.0):
    diffs = np.abs(np.array(user_pitch) - np.array(ref_pitch))
    
    score = 100
    for d in diffs:
        if np.isnan(d):  # pitch 추출 실패한 경우는 감점 안함
            continue
        if d <= threshold:
            continue
        elif d <= threshold + 1:
            score -= 2  # 살짝 큰 오차
        elif d <= threshold + 3:
            score -= 4  # 중간 오차
        else:
            score -= 7  # 심한 오차
    
    return max(0, int(score))

def interpolate_pitch(pitches):
    try:
        if not pitches:
            return []
            
        x = np.arange(len(pitches))
        pitches = np.array(pitches)
        valid = ~np.isnan(pitches)
        
        if np.sum(valid) < 2:
            return pitches.tolist()
            
        f_interp = interp1d(x[valid], pitches[valid], kind='linear', fill_value='extrapolate')
        return f_interp(x).tolist()
    except Exception as e:
        logger.error(f"Pitch 보간 실패: {e}")
        return pitches if isinstance(pitches, list) else pitches.tolist()
    
def safe_convert(value, default):
    try:
        if np.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def result_visualize_syllable(pitch_data, labels, output_dir="media", base64_bool=False):
    """음절 기반 피치 시각화 함수"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("음절 기반 시각화 시작")
        
        # 데이터 유효성 검사
        if not pitch_data or not pitch_data.get('user') or not pitch_data.get('native') or not labels:
            logger.error("pitch_data 또는 labels가 비어있음")
            return None

        def align_data_lengths(data1, data2, label_data):
            min_len = min(len(data1), len(data2), len(label_data))
            return data1[:min_len], data2[:min_len], label_data[:min_len]

        user_pitch, native_pitch, display_labels = align_data_lengths(
            pitch_data['user'], pitch_data['native'], labels
        )
        
        logger.info(f"정렬된 데이터 길이 - user: {len(user_pitch)}, native: {len(native_pitch)}, labels: {len(display_labels)}")
        
        if len(user_pitch) == 0:
            logger.error("정렬된 데이터가 비어있음")
            return None

        # matplotlib 설정 - 한글 폰트 처리
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 그래프 생성
        fig = plt.figure(figsize=(15, 6), facecolor="#A49CC7")
        x = range(len(display_labels))
        
        logger.info(f"그래프 데이터 점 개수: {len(x)}")
        
        # 라인 그래프와 점 표시
        plt.plot(x, native_pitch, label="Native", color="skyblue", linewidth=2, marker='o')
        plt.plot(x, user_pitch, label="User", color="indigo", linewidth=2, marker='o')
        
        # x축 라벨 설정 - 한글 처리
        try:
            # plt.xticks(x, display_labels, fontsize=10, rotation=0)
            plt.xticks(x, display_labels, fontproperties=FontProperties(family='AppleGothic'))
        except Exception as e:
            logger.warning(f"x축 라벨 설정 실패, 인덱스 사용: {e}")
            # plt.xticks(x, [str(i) for i in range(len(display_labels))], fontsize=10)
            plt.xticks(x, [str(i) for i in range(len(display_labels))], fontproperties=FontProperties(family='AppleGothic'))
            
        # 그래프 스타일링
        plt.yticks(color='white')
        plt.title("Pitch Analysis", fontweight='bold', color='white', fontsize=14)
        # plt.xlabel("Syllables", color='white', fontsize=12)
        plt.ylabel("Pitch", fontweight='bold', color='white', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3, color='white')
        plt.tight_layout()
        
        logger.info("그래프 생성 완료, 저장 시작")
        
        # 결과 반환 (Base64 또는 파일 경로)
        if base64_bool:
            logger.info("Base64 인코딩 시작")
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor="#A49CC7")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            logger.info(f"Base64 인코딩 완료, 길이: {len(img_base64)}")
            return img_base64
        else:
            save_path = os.path.join(output_dir, "pitch.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor="#A49CC7")
            plt.close(fig)
            logger.info(f"이미지 파일 저장 완료: {save_path}")
            return os.path.abspath(save_path)
        
    except Exception as e:
        logger.error(f"시각화 오류: {e}")
        traceback.print_exc()
        return None

def run_pitch_analysis(user_denoised_path, ref_denoised_path, user_chunks, ref_chunks):
    """Pitch 분석 함수"""
    try:
        logger.info("=== 피치 분석 핵심 로직 시작 ===")
        logger.info(f"입력 데이터 - User chunks: {len(user_chunks)}, Ref chunks: {len(ref_chunks)}")
        
        # 입력 데이터 유효성 검사
        if not user_chunks or not ref_chunks:
            logger.error("청크 데이터가 비어있음")
            return {"error": "청크 데이터가 비어있음"}
        
        if not os.path.exists(user_denoised_path) or not os.path.exists(ref_denoised_path):
            logger.error("오디오 파일이 존재하지 않음")
            return {"error": "오디오 파일이 존재하지 않음"}

        # 청크 데이터를 타임스탬프 순으로 정렬
        user_chunks_sorted = sorted(user_chunks, key=lambda x: x['timestamp'][0] if x.get('timestamp') else float('inf'))
        ref_chunks_sorted = sorted(ref_chunks, key=lambda x: x['timestamp'][0] if x.get('timestamp') else float('inf'))

        # 음절별 타임스탬프 추출
        logger.info("1. 음절별 타임스탬프 추출 시작")
        user_syllables_data = extract_syllables_from_chunks(user_chunks_sorted)
        ref_syllables_data = extract_syllables_from_chunks(ref_chunks_sorted)

        if not user_syllables_data or not ref_syllables_data:
            logger.error("음절 데이터 추출 실패")
            return {"error": "음절 데이터 추출 실패"}

        # 음절별 피치 추출
        logger.info("2. 음절별 피치 추출 시작")
        user_pitches, user_syllables = extract_pitch_from_syllables(user_denoised_path, user_syllables_data)
        ref_pitches, ref_syllables = extract_pitch_from_syllables(ref_denoised_path, ref_syllables_data)
        
        if not user_pitches or not ref_pitches:
            logger.error("피치 추출 실패")
            return {"error": "피치 추출 실패"}

        # 피치 정규화
        logger.info("3. 피치 정규화 시작")
        user_normalized, ref_normalized = normalize_pitch_for_comparison(user_pitches, ref_pitches)

        # 음절 정렬
        logger.info("4. 음절 정렬 시작")
        user_aligned, ref_aligned, aligned_labels = align_syllables_by_sequence(
            user_syllables, user_normalized, ref_syllables, ref_normalized
        )

        if not user_aligned or not ref_aligned:
            logger.error("음절 정렬 실패")
            return {"error": "음절 정렬 실패"}

        # NaN값 피치 보간
        logger.info("5. 피치 보간 시작")
        user_interpolated = interpolate_pitch(user_aligned)
        ref_interpolated = interpolate_pitch(ref_aligned)
        
        # 유사도 점수 계산
        logger.info("6. 유사도 점수 계산 시작")
        pitch_score = calculate_pitch_score(user_interpolated, ref_interpolated)

        # 결과 시각화
        logger.info("7. 결과 시각화 시작")
        img = result_visualize_syllable(
            pitch_data={
                "user": user_interpolated,     
                "native": ref_interpolated    
            },
            labels=aligned_labels,             
            output_dir="meida",
        )
        
        result = {
            "pitch": {
                "user": [round(safe_convert(p, 0), 2) for p in user_interpolated],
                "native": [round(safe_convert(p, 0), 2) for p in ref_interpolated],
                "labels": aligned_labels,
                "score": int(safe_convert(pitch_score, 0))
            },
            "image": img
        }
        
        logger.info(f"피치 분석 완료 - 점수: {pitch_score}, 정렬된 음절: {len(aligned_labels)}개")
        return result

    except Exception as e:
        logger.error(f"피치 분석 핵심 로직 실패: {e}")
        traceback.print_exc()
        return {"error": f"피치 분석 실패: {str(e)}"}