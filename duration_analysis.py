import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import logging
import traceback
import torch
import os

# MPS 문제 방지
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.backends.mps.is_available = lambda: False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 유효 duration 추출
def get_duration_per_chunk(chunks, audio_file, min_duration=0.05, max_duration=2.0, silence_threshold=0.01):
    # try:
    logger.info("Duration 계산 시작")
    durations = []
    valid_chunks = []
    y, sr = librosa.load(audio_file, sr=None)
    total_audio_duration = len(y) / sr

    frame_length = 2048
    hop_length = 512

    for i, chunk in enumerate(chunks):
        start_time, end_time = chunk['timestamp']
        if end_time is None:
            end_time = total_audio_duration
            chunk['timestamp'] = (start_time, end_time)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        # 첫 단어 or 마지막 단어일 때만 무성음 제거 적용
        if i == 0 or i == len(chunks) - 1:
            rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
            non_silent_frames = np.where(rms > silence_threshold)[0]

            if len(non_silent_frames) > 0:
                first_voice_frame = non_silent_frames[0]
                start_offset = first_voice_frame * hop_length
                segment = segment[start_offset:]
                start_time += start_offset / sr

                rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
                non_silent_frames = np.where(rms > silence_threshold)[0]

                if len(non_silent_frames) > 0:
                    last_voice_frame = non_silent_frames[-1]
                    end_offset = (last_voice_frame + 1) * hop_length
                    segment = segment[:end_offset]
                    end_time = start_time + len(segment) / sr
                else:
                    continue
            else:
                continue

            duration = end_time - start_time
            chunk['timestamp'] = (start_time, end_time)
        else:
            # 중간 단어
            if i + 1 < len(chunks) and chunks[i + 1].get('timestamp'):
                next_start_time, _ = chunks[i + 1]['timestamp']
                if next_start_time > start_time:
                    duration = next_start_time - start_time
                    chunk['timestamp'] = (start_time, next_start_time)
                else:
                    duration = end_time - start_time
                    chunk['timestamp'] = (start_time, end_time)
            else:
                duration = end_time - start_time
                chunk['timestamp'] = (start_time, end_time)

        # duration 검증
        if duration < min_duration or duration > max_duration:
            continue

        durations.append(duration)
        valid_chunks.append(chunk)

    logger.info(f"Duration 계산 완료: {len(durations)}개 값")
    return valid_chunks, durations

# 상대 duration 계산
def get_relative_durations(durations):
    total = sum(durations)
    return [d / total for d in durations] if total > 0 else [0 for _ in durations]


# DTW를 사용한 duration 분석(fastdtw로 수정)
def analyze_duration_with_dtw(user_chunks, ref_chunks, user_durations, ref_durations):
    logger.info("Duration DTW 분석 시작")

    # 길이 확인
    if len(user_durations) == 0 or len(ref_durations) == 0:
        return ["음성 데이터가 부족합니다."], 0, []

    # 상대 duration 계산
    user_rel = np.array(get_relative_durations(user_durations))[:, np.newaxis]
    ref_rel = np.array(get_relative_durations(ref_durations))[:, np.newaxis]

    # fastdtw 실행 (리턴값: distance, path)
    distance, path = fastdtw(user_rel, ref_rel, dist=euclidean)
    x_indices, y_indices = zip(*path)

    # MAE 계산
    abs_diffs = [abs(user_rel[i][0] - ref_rel[j][0]) for i, j in zip(x_indices, y_indices)]
    mae = np.mean(abs_diffs)
    similarity_score = round(max(0, 100 * (1 - mae)))

    # 피드백 생성
    feedback_sentences = []
    highlights = []

    for idx, (i, j) in enumerate(zip(x_indices, y_indices)):
        if i >= len(user_chunks):
            break

        u = user_rel[i][0]
        r = ref_rel[j][0]
        word = user_chunks[i].get('text', '')
        diff = u - r

        highlights.append(abs(diff) > 0.1)

        if diff > 0.1:
            feedback_sentences.append(f"'{word}' 단어를 상대적으로 길게 발음했습니다.")
        elif diff < -0.1:
            feedback_sentences.append(f"'{word}' 단어를 상대적으로 짧게 발음했습니다.")

    if not feedback_sentences:
        feedback_sentences.append("모든 단어의 발화 길이가 적절했습니다.")

    logger.info(f"Duration DTW 분석 완료: 점수 {similarity_score}")
    return feedback_sentences, similarity_score, highlights

def run_duration_analysis(user_denoised_path, ref_denoised_path, user_chunks, ref_chunks):
    """duration 분석 함수"""
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

        logger.info("Duration 분석 시작")
        user_chunks_filtered, user_durations = get_duration_per_chunk(user_chunks_sorted, user_denoised_path)
        ref_chunks_filtered, ref_durations = get_duration_per_chunk(ref_chunks_sorted, ref_denoised_path)

        duration_feedbacks, duration_score, duration_highlights = analyze_duration_with_dtw(
            user_chunks_filtered, ref_chunks_filtered, user_durations, ref_durations
        )

        def safe_convert(value, default=None):
            try:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return default
                return float(value) if isinstance(value, (int, float, np.number)) else value
            except:
                return default

        result = {
            "duration": {
                "user": [round(safe_convert(d, 0), 2) for d in user_durations],
                "native": [round(safe_convert(d, 0), 2) for d in ref_durations],
                "score": int(safe_convert(duration_score, 0)),
                "highlight": [bool(h) for h in duration_highlights[:len(user_durations)]] if duration_highlights else [],
                "feedback": list(duration_feedbacks)
            }
        }
        
        logger.info(f"Duration 분석 완료 - 점수: {duration_score}")
        return result

    except Exception as e:
        logger.error(f"Duration 분석 실패: {e}")
        traceback.print_exc()
        return {"error": f"Duration 분석 실패: {str(e)}"}
