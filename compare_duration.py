import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline
from dtw import accelerated_dtw


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
        print(f"Saved denoised audio to: {output_file}")

        return output_file
    except Exception as e:
        print(f"Error during spectral subtraction: {e}")
        import traceback
        traceback.print_exc()
        return None


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

# 유효 duration 추출
def get_duration_per_chunk(chunks, audio_file, min_duration=0.05, max_duration=2.0):
    durations = []
    valid_chunks = []
    y, sr = librosa.load(audio_file, sr=None)
    total_audio_duration = len(y) / sr
    for chunk in chunks:
        start_time, end_time = chunk['timestamp']
        if end_time is None:
            end_time = total_audio_duration
            chunk['timestamp'][1] = end_time
        duration = end_time - start_time
        if duration < min_duration or duration > max_duration:
            continue
        durations.append(duration)
        valid_chunks.append(chunk)
    return valid_chunks, durations

# 상대 duration 계산
def get_relative_durations(durations):
    total = sum(durations)
    return [d / total for d in durations] if total > 0 else [0 for _ in durations]

# 실행
def run_dtw_analysis(user_audio, ref_audio):
    user_denoised = spectral_subtraction(user_audio, "user_summer_denoised.wav")
    ref_denoised = spectral_subtraction(ref_audio, "summer_denoised.wav")
    if not user_denoised or not ref_denoised:
        return

    user_result = transcribe_with_whisper(user_denoised)
    ref_result = transcribe_with_whisper(ref_denoised)
    user_chunks = user_result['chunks']
    ref_chunks = ref_result['chunks']

    user_chunks, user_durations = get_duration_per_chunk(user_chunks, user_denoised)
    ref_chunks, ref_durations = get_duration_per_chunk(ref_chunks, ref_denoised)

    print("\n사용자 단어별 구간:")
    for chunk in user_chunks:
        if chunk['timestamp']:
            print(f"{chunk['text']}: {chunk['timestamp'][0]:.2f}s ~ {chunk['timestamp'][1]:.2f}s")

    print("\n모범 단어별 구간:")
    for chunk in ref_chunks:
        if chunk['timestamp']:
            print(f"{chunk['text']}: {chunk['timestamp'][0]:.2f}s ~ {chunk['timestamp'][1]:.2f}s")

    if len(user_durations) == 0 or len(ref_durations) == 0:
        print("비교할 유효한 duration이 부족합니다.")
        return

    # 상대 duration
    user_rel = np.array(get_relative_durations(user_durations))[:, np.newaxis]
    ref_rel = np.array(get_relative_durations(ref_durations))[:, np.newaxis]

    # DTW 수행
    dist, cost, acc_cost, path = accelerated_dtw(
        user_rel,
        ref_rel,
        dist='euclidean'
    )
    dist, cost, acc_cost, path = accelerated_dtw(user_rel, ref_rel, dist='euclidean')
    similarity_score = round(max(0, 100 * (1 - dist)))

    feedback_sentences = []

    print("\n[단어별 상대 길이 비교]")
    for (i, j) in zip(*path):
        u = user_rel[i][0]
        r = ref_rel[j][0]
        word = user_chunks[i]['text']
        diff = u - r
        percent_diff = diff * 100

        user_dur = user_durations[i]
        ref_dur = ref_durations[j]

        feedback_line = f"{word:10s}| 사용자: {user_dur:.2f}s / 모범: {ref_dur:.2f}s | {percent_diff:+.1f}%"

        if diff > 0.1:
            
            feedback_sentences.append(f"'{word}' 단어를 상대적으로 길게 발음했습니다.")
        elif diff < -0.1:
            
            feedback_sentences.append(f"'{word}' 단어를 상대적으로 짧게 발음했습니다.")


        print(feedback_line)

    # 피드백 요약 출력
    print("\n피드백:")
    if feedback_sentences:
        for sent in feedback_sentences:
            print("-", sent)
    else:
        print("모든 단어의 상대 발화 길이가 적절했습니다.")

    # 점수 출력
    print(f"\n종합 점수: {similarity_score}점 ")


run_dtw_analysis("user_summer.wav", "summer.wav")
