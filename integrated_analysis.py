 from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from matplotlib.font_manager import FontProperties
from g2pk import G2p
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

# G2p 전역 변수로 한 번만 초기화
try:
    g2p = G2p()
    logger.info("G2p 초기화 완료")
except Exception as e:
    logger.error(f"G2p 초기화 실패: {e}")
    g2p = None

# 노이즈 제거 함수
def spectral_subtraction(audio_file, output_file, alpha=2.0, beta=0.05):
    y, sr = librosa.load(audio_file, sr=None)
    frame_length = 1024
    hop_length = 512
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, win_length=frame_length)
    magnitude, phase = np.abs(D), np.angle(D)
    power = magnitude**2
    noise_frames = min(int(0.5 * sr / hop_length), 10)
    noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
    gain = np.maximum(1 - alpha * (noise_power / (power + 1e-10)), beta)
    enhanced_magnitude = magnitude * gain
    enhanced_D = enhanced_magnitude * np.exp(1j * phase)
    enhanced_y = librosa.istft(enhanced_D, hop_length=hop_length, win_length=frame_length)

    if len(enhanced_y) > len(y):
        enhanced_y = enhanced_y[:len(y)]
    elif len(enhanced_y) < len(y):
        enhanced_y = np.pad(enhanced_y, (0, len(y) - len(enhanced_y)))

    sf.write(output_file, enhanced_y, sr)
    logger.info(f"노이즈 제거 완료: {output_file}")
    return output_file

# Whisper로 텍스트 및 타임스탬프 추출
def transcribe_with_whisper(audio_file):
    # try:
    logger.info(f"Whisper 전사 시작: {audio_file}")
    pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        return_timestamps="word",
        device="cpu" ,  # 명시적으로 CPU 사용
        torch_dtype=torch.float32
    )
    result = pipe(audio_file)

    if not result or 'chunks' not in result:
        logger.error("Whisper 결과가 비어있거나 chunks가 없음")
        return None

    duration = librosa.get_duration(path=audio_file)

    for chunk in result['chunks']:
        if chunk.get('timestamp'):
            start, end = chunk['timestamp']
            if end is None:
                end = duration
            chunk['timestamp'] = (round(start, 2), round(end, 2))
        else:
            logger.warning(f"타임스탬프가 없는 청크 발견: {chunk}")

    logger.info(f"Whisper 전사 완료: {len(result['chunks'])}개 청크")
    return result

# pitch 추출을 위한 음소 단위 분해
def extract_phonemes_from_chunks(chunks):
    try:
        logger.info("음소 추출 시작")
        if not g2p:
            logger.error("G2p가 초기화되지 않음")
            return []
        
        phonemes = []
        for chunk in chunks:                
            word = chunk["text"].strip()
            start, end = chunk["timestamp"]
            duration = end - start
            
            if duration <= 0:
                continue
                
            try:
                # g2pk로 음소 변환
                chars = list(g2p(word))
                if not chars:
                    continue
                    
                step = duration / len(chars)
                for i, c in enumerate(chars):
                    phonemes.append({
                        "text": c,
                        "timestamp": [start + i * step, start + (i + 1) * step]
                    })
            except Exception as e:
                logger.warning(f"음소 변환 실패 for '{word}': {e}")
                continue
                
        logger.info(f"음소 추출 완료: {len(phonemes)}개 음소")
        return phonemes
    except Exception as e:
        logger.error(f"음소 추출 실패: {e}")
        return []

# 음소 단위 pitch 추출
def extract_pitch_sequence(audio_path, phonemes):
    try:
        logger.info("Pitch 추출 시작")
        y, sr = librosa.load(audio_path, sr=None)
        f0, _, _ = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            sr=sr
        )
        times = librosa.times_like(f0, sr=sr)
        pitches = []
        labels = []
        
        for ph in phonemes:
            if not ph.get('timestamp'):
                continue
                
            start, end = ph['timestamp']
            start_idx = np.argmin(np.abs(times - start))
            end_idx = np.argmin(np.abs(times - end))
            segment = f0[start_idx:end_idx+1]
            segment = segment[~np.isnan(segment)]
            pitch = np.median(segment) if len(segment) > 0 else np.nan
            pitches.append(pitch)
            labels.append(ph['text'])
            
        logger.info(f"Pitch 추출 완료: {len(pitches)}개 pitch 값")
        return pitches, labels
    except Exception as e:
        logger.error(f"Pitch 추출 실패: {e}")
        return [], []

# 발음 기반 음성 보간
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

def align_pitch_by_phoneme(native_pitch, native_labels, user_pitch, user_labels):
    try:
        matched_native = []
        matched_user = []
        matched_labels = []
        native_dict = {label: pitch for label, pitch in zip(native_labels, native_pitch)}
        
        for u_label, u_pitch in zip(user_labels, user_pitch):
            if u_label in native_dict:
                matched_labels.append(u_label)
                matched_native.append(native_dict[u_label])
                matched_user.append(u_pitch)
                
        logger.info(f"Pitch 정렬 완료: {len(matched_user)}개 매칭")
        return matched_user, matched_native, matched_labels
    except Exception as e:
        logger.error(f"Pitch 정렬 실패: {e}")
        return [], [], []

# 단어별 intensity 계산
def get_intensity_per_chunk(audio_file, chunks):
    # try:
    logger.info("Intensity 계산 시작")
    y, sr = librosa.load(audio_file, sr=None)
    intensities = []
    words = []
    
    for chunk in chunks:
        # if not chunk.get('timestamp') or not chunk.get('text'):
        #     continue
            
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

# 강세 비교 분석
def compare_stress(words_ref, intensities_ref, words_usr, intensities_usr):
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

# DTW를 사용한 duration 분석(fastdtw로 수정)
def analyze_duration_with_dtw(user_chunks, ref_chunks, user_durations, ref_durations):
    logger.info("Duration DTW 분석 시작")

    # 길이 확인
    if len(user_durations) == 0 or len(ref_durations) == 0:
        return ["음성 데이터가 부족합니다."], 0, []

    # 상대 duration 계산
    user_rel = np.array(get_relative_durations(user_durations))[:, np.newaxis]
    ref_rel = np.array(get_relative_durations(ref_durations))[:, np.newaxis]

    # 길이가 너무 짧으면 DTW 불가
    if len(user_rel) < 2 or len(ref_rel) < 2:
        return ["Duration 시퀀스 길이가 너무 짧아 DTW 분석이 어렵습니다."], 0, []

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

# 시각화
def result_visualize(pitch_data, labels, output_dir="pitch_result"):
    os.makedirs(output_dir, exist_ok=True)

    def align_data_lengths(data1, data2, label_data):
        min_len = min(len(data1), len(data2), len(label_data))
        return data1[:min_len], data2[:min_len], label_data[:min_len]

    # Pitch Plot (음소 단위)
    if pitch_data['user'] and pitch_data['native'] and labels:
        user_pitch, native_pitch, pitch_labels = align_data_lengths(
            pitch_data['user'], pitch_data['native'], labels
        )

        fig = plt.figure(figsize=(10, 5), facecolor="#A49CC7")
        x = range(len(pitch_labels))
        plt.plot(x, native_pitch, label="Native", color="skyblue", linewidth=2)
        plt.plot(x, user_pitch, label="User", color="indigo", linewidth=2)
        plt.scatter(x, native_pitch, color='skyblue', edgecolors='skyblue', zorder=5)
        plt.scatter(x, user_pitch, color='indigo', edgecolors='indigo', zorder=5)
        plt.xticks(x, pitch_labels, fontproperties=FontProperties(family='AppleGothic'))
        plt.yticks(color='white')
        plt.title("Pitch Analysis", fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pitch.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

# 통합 분석 실행 함수
def run_integrated_analysis(user_audio, ref_audio):
    try:
        logger.info(f"=== 통합 분석 시작 ===")
        logger.info(f"User: {user_audio}, Ref: {ref_audio}")
        
        # 노이즈 제거
        logger.info("1. 노이즈 제거 시작")
        user_denoised = spectral_subtraction(user_audio, "user_denoised.wav")
        ref_denoised = spectral_subtraction(ref_audio, "ref_denoised.wav")
        
        if not user_denoised or not ref_denoised:
            logger.error("노이즈 제거 실패")
            return {"error": "노이즈 제거 실패"}

        # 음성 인식
        logger.info("2. 음성 인식 시작")
        user_result = transcribe_with_whisper(user_denoised)
        ref_result = transcribe_with_whisper(ref_denoised)

        if not user_result or not ref_result:
            logger.error("음성 인식 실패")
            return {"error": "음성 인식 실패"}

        user_chunks = sorted(user_result['chunks'], key=lambda x: x['timestamp'][0] if x.get('timestamp') else float('inf'))
        ref_chunks = sorted(ref_result['chunks'], key=lambda x: x['timestamp'][0] if x.get('timestamp') else float('inf'))

        # 음소 추출 및 pitch 분석
        logger.info("3. 음소 추출 및 pitch 분석 시작")
        user_phonemes = extract_phonemes_from_chunks(user_chunks)
        ref_phonemes = extract_phonemes_from_chunks(ref_chunks)

        user_pitch, user_labels = extract_pitch_sequence(user_denoised, user_phonemes)
        ref_pitch, ref_labels = extract_pitch_sequence(ref_denoised, ref_phonemes)
        
        user_pitch_interp = interpolate_pitch(user_pitch)
        ref_pitch_interp = interpolate_pitch(ref_pitch)     
        
        user_pitch_aligned, ref_pitch_aligned, aligned_labels = align_pitch_by_phoneme(
            ref_pitch_interp, ref_labels, user_pitch_interp, user_labels
        )

        # Pitch 점수 계산
        if len(user_pitch_aligned) > 0 and len(ref_pitch_aligned) > 0:
            pitch_diff = np.array(user_pitch_aligned) - np.array(ref_pitch_aligned)
            pitch_score = max(0, int(100 - np.nanmean(np.abs(pitch_diff)) / 10))
        else:
            pitch_score = 0

        # Intensity 분석
        logger.info("4. Intensity 분석 시작")
        words_usr, intensities_usr = get_intensity_per_chunk(user_denoised, user_chunks)
        words_ref, intensities_ref = get_intensity_per_chunk(ref_denoised, ref_chunks)

        usr_z, ref_z, stress_score, stress_highlights, stress_feedbacks = compare_stress(
            words_ref, intensities_ref, words_usr, intensities_usr
        )

        # Duration 분석
        logger.info("5. Duration 분석 시작")
        user_chunks_filtered, user_durations = get_duration_per_chunk(user_chunks, user_denoised)
        ref_chunks_filtered, ref_durations = get_duration_per_chunk(ref_chunks, ref_denoised)

        duration_feedbacks, duration_score, duration_highlights = analyze_duration_with_dtw(
            user_chunks_filtered, ref_chunks_filtered, user_durations, ref_durations
        )

        # 안전한 결과 생성
        logger.info("6. 결과 생성 시작")
        def safe_convert(value, default=None):
            try:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return default
                return float(value) if isinstance(value, (int, float, np.number)) else value
            except:
                return default

        # pitch 시각화 추가
        result_visualize(
            pitch_data={
                "user": user_pitch_aligned,
                "native": ref_pitch_aligned
            },
            labels=aligned_labels,
            output_dir="pitch_result"
        )

        result = {
            "pitch": {
                "user": [round(safe_convert(p, 0), 2) for p in user_pitch_aligned],
                "native": [round(safe_convert(p, 0), 2) for p in ref_pitch_aligned],
                "score": int(safe_convert(pitch_score, 0))
            },
            "intensity": {
                "user": [round(safe_convert(z, 0), 2) for z in usr_z],
                "native": [round(safe_convert(z, 0), 2) for z in ref_z],
                "score": int(safe_convert(stress_score, 0)),
                "highlight": [bool(h) for h in stress_highlights],
                "feedback": list(stress_feedbacks)
            },
            "duration": {
                "user": [round(safe_convert(d, 0), 2) for d in user_durations],
                "native": [round(safe_convert(d, 0), 2) for d in ref_durations],
                "score": int(safe_convert(duration_score, 0)),
                "highlight": [bool(h) for h in duration_highlights[:len(user_durations)]] if duration_highlights else [],
                "feedback": list(duration_feedbacks)
            }
        }

        logger.info("=== 통합 분석 완료 ===")
        return result

    except Exception as e:
        logger.error(f"통합 분석 중 예상치 못한 에러: {e}")
        traceback.print_exc()
        return {"error": f"분석 실패: {str(e)}"}