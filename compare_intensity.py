from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re

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

    # 오디오 길이 얻기
    duration = librosa.get_duration(filename=audio_file)

    # 마지막 단어의 end=None인 경우 보정
    for chunk in result['chunks']:
        start, end = chunk['timestamp']
        if end is None:
            end = duration
            
        chunk['timestamp'] = (round(start, 2), round(end, 2))

    return result


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


def print_intensity_report(title, words, intensities):
    avg = np.mean(intensities)
    z_scores = zscore(intensities)
    print(f"\n평균 Intensity: {avg:.5f}")
    print("단어별 Intensity 및 상대 강세 (z-score):\n")

    for word, intensity, z in zip(words, intensities, z_scores):
        print(f"{word:10s} | Intensity: {intensity:.5f} | z-score: {z:+.2f}")



def visualize_user_emphasis(words_usr, feedbacks):
    # 한글 폰트 설정
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 피드백에서 언급된 단어 추출
    emphasized_words = []
    deemphasized_words = []
    
    for feedback in feedbacks:
        word_match = re.search(r"'([^']+)'", feedback)
        if word_match:
            target = word_match.group(1).strip()
            for word in words_usr:
                if target in word:
                    if "불필요한 강조" in feedback:
                        emphasized_words.append(word)
                    elif "약하게 발음" in feedback:
                        deemphasized_words.append(word)

    # 그림 생성
    plt.figure(figsize=(6, 3))
    plt.xlim(0, len(words_usr))  
    plt.ylim(0, 1)  
    plt.axis('off')

    # 단어 표시
    for i, word in enumerate(words_usr):
        color = 'black'
        fontweight = 'normal'
        
        if word in emphasized_words:
            color = 'red'
            fontweight = 'bold'
            
        elif word in deemphasized_words:
            color = 'gray' 
            fontweight = 'light'
        
        plt.text(i, 0.7, word,
                fontproperties=font_prop,
                color=color,
                fontweight=fontweight,
                fontsize=24,
                ha='center')
    
        
        # 피드백 텍스트 추가
        if feedbacks:
            feedback_text = []
            for feedback in feedbacks:
                feedback_text.append(f"- {feedback}")
            
            feedback_combined = '\n'.join(feedback_text)
            plt.figtext(0.5, 0.3, feedback_combined, 
                    fontproperties=font_prop,
                    fontsize=14,
                    ha='center', 
                    bbox={'facecolor':'whitesmoke', 'edgecolor':'lightgray', 'alpha':0.8, 'pad':10})
        
    plt.axis('off')
    
    # 여백 조정
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.tight_layout()
    
    # 그래프 저장 및 표시
    plt.savefig('emphasis_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_and_generate_feedback(words_ref, intensities_ref, words_usr, intensities_usr):
    feedbacks = []
    total_score = 0
    per_word_scores = []

    ref_z = (intensities_ref - np.mean(intensities_ref)) / np.std(intensities_ref) if len(intensities_ref) > 1 else np.zeros_like(intensities_ref)
    usr_z = (intensities_usr - np.mean(intensities_usr)) / np.std(intensities_usr) if len(intensities_usr) > 1 else np.zeros_like(intensities_usr)

    for i in range(min(len(words_ref), len(words_usr))):
        usr_w = words_usr[i]
        diff = usr_z[i] - ref_z[i]

        # 점수 계산
        score = max(0.0, 1.0 - abs(diff) / 2.0)
        score = int(round(score * 100))
        per_word_scores.append(score)
        total_score += score

        if diff > 1.0:
            feedbacks.append(f"'{usr_w}' 단어에 불필요한 강조가 있습니다.")
        elif diff < -1.0:
            feedbacks.append(f"'{usr_w}' 단어가 약하게 발음되었습니다.")

    avg_score = int(round(total_score / len(per_word_scores))) if per_word_scores else 0

    if avg_score >= 85 and not feedbacks:
        feedbacks.append("전반적으로 강세를 매우 잘 따라했습니다!")
    elif avg_score >= 75 and not feedbacks:
        feedbacks.append("강세 전달이 자연스럽고 좋습니다!")

    return feedbacks, avg_score


# === 실행 ===

# 모범 발화
ref_input = "summer.wav"
ref_denoised = "summer_denoised.wav"
ref_clean = spectral_subtraction(ref_input, ref_denoised)
ref_result = transcribe_with_whisper(ref_clean)
ref_chunks = sorted(ref_result['chunks'], key=lambda x: (x['timestamp'] if x['timestamp'] else [float('inf')]))
words_ref, intensities_ref = get_intensity_per_chunk(ref_clean, ref_chunks)

# 학습자 발화
usr_input = "user_summer.wav"
usr_denoised = "user_summer_denoised.wav"
usr_clean = spectral_subtraction(usr_input, usr_denoised)
usr_result = transcribe_with_whisper(usr_clean)
usr_chunks = sorted(usr_result['chunks'], key=lambda x: (x['timestamp'] if x['timestamp'] else [float('inf')]))
words_usr, intensities_usr = get_intensity_per_chunk(usr_clean, usr_chunks)

# 리포트 출력
print_intensity_report("모범 발화 분석", words_ref, intensities_ref)
print_intensity_report("학습자 발화 분석", words_usr, intensities_usr)

feedbacks, avg_score = compare_and_generate_feedback(words_ref, intensities_ref, words_usr, intensities_usr)

# 출력
print("\n피드백:")
for f in feedbacks:
    print("-", f)
print(f"\n종합 점수: {avg_score}점")
    
# 시각화
visualize_user_emphasis(words_usr, feedbacks)