from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from dtw import accelerated_dtw

# 노이즈 제거 함수
def spectral_subtraction(audio_file, output_file, alpha=2.0, beta=0.05):
    try:
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
        return output_file
    except Exception as e:
        print(f"Error during spectral subtraction: {e}")
        return None

# Whisper로 텍스트 및 타임스탬프 추출
def transcribe_with_whisper(audio_file):
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", return_timestamps="word")
    result = pipe(audio_file)

    duration = librosa.get_duration(path=audio_file)

    for chunk in result['chunks']:
        start, end = chunk['timestamp']
        if end is None:
            end = duration
        chunk['timestamp'] = (round(start, 2), round(end, 2))

    return result

# 단어별 intensity 계산
def get_intensity_per_chunk(audio_file, chunks):
    y, sr = librosa.load(audio_file, sr=None)
    intensities = []
    words = []
    for chunk in chunks:
        if chunk['timestamp'] is None or chunk['timestamp'][0] is None or chunk['timestamp'][1] is None:
            continue
        start_time, end_time = chunk['timestamp']
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        rms = np.sqrt(np.mean(segment**2)) if len(segment) > 0 else 0.0
        intensities.append(rms)
        words.append(chunk['text'])
    return words, intensities

# 유효 duration 추출
def get_duration_per_chunk(chunks, audio_file, min_duration=0.05, max_duration=2.0, silence_threshold=0.01):
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

        # === 첫 단어 or 마지막 단어일 때만 무성음 제거 적용 ===
        if i == 0 or i == len(chunks) - 1:
            # 앞부분 무성음 제거
            rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
            non_silent_frames = np.where(rms > silence_threshold)[0]

            if len(non_silent_frames) > 0:
                first_voice_frame = non_silent_frames[0]
                start_offset = first_voice_frame * hop_length
                segment = segment[start_offset:]
                start_time += start_offset / sr

                # 뒷부분 무성음 제거
                rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
                non_silent_frames = np.where(rms > silence_threshold)[0]

                if len(non_silent_frames) > 0:
                    last_voice_frame = non_silent_frames[-1]
                    end_offset = (last_voice_frame + 1) * hop_length
                    segment = segment[:end_offset]
                    end_time = start_time + len(segment) / sr
                else:
                    continue  # 전체 silence면 skip
            else:
                continue  # 전체 silence면 skip

            # duration 계산
            duration = end_time - start_time
            chunk['timestamp'] = (start_time, end_time)

        else:
            # 중간 단어 → pause 포함 (무성음 제거 X)
            next_start_time, _ = chunks[i + 1]['timestamp']
            if next_start_time > start_time:
                duration = next_start_time - start_time
                chunk['timestamp'] = (start_time, next_start_time)
            else:
                # fallback 처리
                duration = end_time - start_time
                chunk['timestamp'] = (start_time, end_time)

        # duration 검증
        if duration < min_duration or duration > max_duration:
            continue

        durations.append(duration)
        valid_chunks.append(chunk)

    return valid_chunks, durations

# 상대 duration 계산
def get_relative_durations(durations):
    total = sum(durations)
    return [d / total for d in durations] if total > 0 else [0 for _ in durations]

# 강세 비교 분석
def compare_stress(words_ref, intensities_ref, words_usr, intensities_usr):
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

    return usr_z, ref_z, avg_score, highlights, feedbacks

# DTW를 사용한 duration 분석
def analyze_duration_with_dtw(user_chunks, ref_chunks, user_durations, ref_durations):
    if len(user_durations) == 0 or len(ref_durations) == 0:
        return [], 0, []

    user_rel = np.array(get_relative_durations(user_durations))[:, np.newaxis]
    ref_rel = np.array(get_relative_durations(ref_durations))[:, np.newaxis]

    dist, cost, acc_cost, path = accelerated_dtw(user_rel, ref_rel, dist='euclidean')

    abs_diffs = [abs(user_rel[i][0] - ref_rel[j][0]) for i, j in zip(*path)]
    mae = np.mean(abs_diffs)
    similarity_score = round(max(0, 100 * (1 - mae)))

    feedback_sentences = []
    highlights = []

    for idx, (i, j) in enumerate(zip(*path)):
        if idx >= len(user_chunks):
            break
        
        u = user_rel[i][0]
        r = ref_rel[j][0]
        word = user_chunks[i]['text']
        diff = u - r

        highlights.append(abs(diff) > 0.1)

        if diff > 0.1:
            feedback_sentences.append(f"'{word}' 단어를 상대적으로 길게 발음했습니다.")
        elif diff < -0.1:
            feedback_sentences.append(f"'{word}' 단어를 상대적으로 짧게 발음했습니다.")

    if not feedback_sentences:
        feedback_sentences.append("모든 단어의 발화 길이가 적절했습니다.")

    return feedback_sentences, similarity_score, highlights


# 통합 분석 실행 함수
def run_integrated_analysis(user_audio, ref_audio):
    # 1. 노이즈 제거
    user_denoised = spectral_subtraction(user_audio, "user_denoised.wav")
    ref_denoised = spectral_subtraction(ref_audio, "ref_denoised.wav")
    
    if not user_denoised or not ref_denoised:
        print("노이즈 제거 실패")
        return None

    # 2. 음성 인식
    user_result = transcribe_with_whisper(user_denoised)
    ref_result = transcribe_with_whisper(ref_denoised)

    user_chunks = sorted(user_result['chunks'], key=lambda x: x['timestamp'][0] if x['timestamp'] else float('inf'))
    ref_chunks = sorted(ref_result['chunks'], key=lambda x: x['timestamp'][0] if x['timestamp'] else float('inf'))

    # 3. Intensity 분석
    words_usr, intensities_usr = get_intensity_per_chunk(user_denoised, user_chunks)
    words_ref, intensities_ref = get_intensity_per_chunk(ref_denoised, ref_chunks)

    usr_z, ref_z, stress_score, stress_highlights, stress_feedbacks = compare_stress(
        words_ref, intensities_ref, words_usr, intensities_usr
    )

    # 4. Duration 분석
    user_chunks_filtered, user_durations = get_duration_per_chunk(user_chunks, user_denoised)
    ref_chunks_filtered, ref_durations = get_duration_per_chunk(ref_chunks, ref_denoised)

    duration_feedbacks, duration_score, duration_highlights = analyze_duration_with_dtw(
        user_chunks_filtered, ref_chunks_filtered, user_durations, ref_durations
    )


    # 5. JSON 결과 반환
    result = {
        "intensity": {
            "user": [round(float(z), 2) for z in usr_z],
            "native": [round(float(z), 2) for z in ref_z],
            "score": int(stress_score),
            "highlight": [bool(h) for h in stress_highlights],
            "feedback": list(stress_feedbacks)
        },
        "duration": {
            "user": [round(float(d), 2) for d in user_durations],
            "native": [round(float(d), 2) for d in ref_durations],
            "score": int(duration_score),
            "highlight": [bool(h) for h in duration_highlights[:len(user_durations)]] if duration_highlights else [],
            "feedback": list(duration_feedbacks)
        }
    }

    return result

