import os
import whisper
import torch
import subprocess
import traceback
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate, interp1d
from glob import glob
from matplotlib.font_manager import FontProperties
from g2pk import G2p

# G2P 객체
g2p = G2p()

def convert_to_wav(input_file, output_file=None, sample_rate=16000):
    try:
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{base_name}.wav"
        if os.path.abspath(input_file) == os.path.abspath(output_file):
            print("입력과 출력 경로가 같아서 변환을 생략합니다.")
            return input_file
        command = ["ffmpeg", "-i", input_file, "-ar", str(sample_rate), "-ac", "1", "-y", output_file]
        subprocess.run(command, check=True)
        return output_file
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

def spectral_subtraction(audio_file, output_file, alpha=2.0, beta=0.05):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        frame_length = 1024
        hop_length = 512
        D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, win_length=frame_length)
        mag, phase = np.abs(D), np.angle(D)
        power = mag**2
        noise_frames = min(int(0.5 * sr / hop_length), 10)
        noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
        gain = np.maximum(1 - alpha * (noise_power / (power + 1e-10)), beta)
        enhanced_mag = mag * gain
        enhanced_D = enhanced_mag * np.exp(1j * phase)
        enhanced_y = librosa.istft(enhanced_D, hop_length=hop_length, win_length=frame_length)
        enhanced_y = np.pad(enhanced_y, (0, max(0, len(y) - len(enhanced_y))))[:len(y)]
        sf.write(output_file, enhanced_y, sr)
        return output_file
    except Exception as e:
        print(f"Error in spectral subtraction: {e}")
        return None

def extract_phonemes(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="ko", word_timestamps=True)
    phonemes = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            word = word_info["word"].strip()
            start, end = word_info["start"], word_info["end"]
            duration = end - start
            chars = list(word)
            step = duration / max(1, len(chars))
            for i, c in enumerate(chars):
                phonemes.append({"text": c, "timestamp": [start + i * step, start + (i + 1) * step]})
    return phonemes

def extract_pitch_sequence(audio_path, phonemes):
    y, sr = librosa.load(audio_path, sr=None)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    times = librosa.times_like(f0, sr=sr)
    pitches, labels = [], []
    for ph in phonemes:
        start, end = ph['timestamp']
        start_idx = np.argmin(np.abs(times - start))
        end_idx = np.argmin(np.abs(times - end))
        segment = f0[start_idx:end_idx+1]
        segment = segment[~np.isnan(segment)]
        pitch = np.median(segment) if len(segment) > 0 else np.nan
        pitches.append(pitch)
        labels.append(ph['text'])
    return pitches, labels

def interpolate_pitch(pitches):
    x = np.arange(len(pitches))
    pitches = np.array(pitches)
    valid = ~np.isnan(pitches)
    if np.sum(valid) < 2:
        return pitches
    f_interp = interp1d(x[valid], pitches[valid], kind='linear', fill_value='extrapolate')
    return f_interp(x)

def align_pitch_by_phoneme(native_pitch, native_labels, user_pitch, user_labels):
    matched_native = []
    matched_user = []
    matched_labels = []
    native_dict = {label: pitch for label, pitch in zip(native_labels, native_pitch)}
    for u_label, u_pitch in zip(user_labels, user_pitch):
        if u_label in native_dict:
            matched_labels.append(u_label)
            matched_native.append(native_dict[u_label])
            matched_user.append(u_pitch)
    return matched_user, matched_native, matched_labels

def plot_pitch_comparison(user_pitch, native_pitch, labels, output_path="visualization/korean_transcriber_matched.png"):
    x = np.linspace(0, 1, len(labels))
    plt.figure(figsize=(10, 6), facecolor="#A49CC7")
    plt.plot(x, native_pitch, label="Native", color="skyblue", linewidth=2)
    plt.plot(x, user_pitch, label="User", color="indigo", linewidth=2)
    plt.scatter(x, native_pitch, color='skyblue', edgecolors='black', zorder=5)
    plt.scatter(x, user_pitch, color='indigo', edgecolors='black', zorder=5)
    plt.xticks(x, labels, fontproperties=FontProperties(family='AppleGothic'))
    plt.yticks(color='white')
    plt.title("Pitch Camparison", color="white", fontsize=14, fontweight='bold', loc='left')
    plt.legend(loc="upper right", frameon=False, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor="#A49CC7")
    plt.show()

def select_contents_directory():
    """ 컨텐츠 디렉토리 선택 """
    # 사용 가능한 컨텐츠 디렉토리 목록
    available_contents = ['misaeng', 'our_beloved_summer', 'cheese_in_the_trap', 'stove_league']
    
    print("\n=== Select Content===")
    print("Select the content folder you want to process:")
    
    for idx, content in enumerate(available_contents, 1):
        print(f"{idx}. {content}")
    
    while True:
        try:
            choice = input("Enter number (1-4): ").strip()
            choice_idx = int(choice) - 1

            if 0 <= choice_idx < len(available_contents):
                return available_contents[choice_idx]
            else:
                print("Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number or drama name.")

def select_file(contents_dir, user_filename=None):
    """ 디렉토리 내 파일 선택 """
    # 디렉토리 경로
    content_path = os.path.join("mp4_clip", contents_dir)
  
    if not os.path.exists(content_path):
        print(f"Error: '{content_path}' 디렉토리를 찾을 수 없습니다.")
        return None
    
    # 디렉토리 내의 모든 mp4 파일 찾기
    mp4_files = glob(os.path.join(content_path, "*.mp4"))
    
    if not mp4_files:
        print(f"오류: 'Could not find directory '{content_path}'.")
        return None
    
    if user_filename:
        # 확장자가 없으면 추가
        if not user_filename.lower().endswith('.mp4'):
            user_filename += '.mp4'
            
        # 지정된 파일명과 일치하는 파일 찾기
        matching_files = [f for f in mp4_files if os.path.basename(f) == user_filename]
        
        if matching_files:
            selected_file = matching_files[0]
            print(f"The specified file was found:{os.path.basename(selected_file)}")
            return selected_file
        else:
            print(f"Error: File '{user_filename}' not found.")
            return None

def main():
    drama_dir = select_contents_directory()
    drama_filename = input("\nEnter the drama file name to process (without extension): ").strip()
    drama_file = select_file(drama_dir, drama_filename)
    if not drama_file:
        print("The file to process could not be found. The program will terminate.")
        return
    user_input = input("사용자 음성 파일(wav): ").strip()
    base_native = os.path.splitext(os.path.basename(drama_file))[0]
    native_wav = convert_to_wav(drama_file, f"wav_file/{base_native}.wav")
    native_clean = spectral_subtraction(native_wav, f"denoised_audio/{base_native}_denoised.wav")
    native_phonemes = extract_phonemes(native_clean)
    user_phonemes = extract_phonemes(user_input)

    print("\nNative phonemes:")
    print([p["text"] for p in native_phonemes])
    print("\nUser phonemes:")
    print([p["text"] for p in user_phonemes])

    native_pitch, native_labels = extract_pitch_sequence(native_clean, native_phonemes)
    user_pitch, user_labels = extract_pitch_sequence(user_input, user_phonemes)
    user_pitch, native_pitch, labels = align_pitch_by_phoneme(native_pitch, native_labels, user_pitch, user_labels)
    plot_pitch_comparison(user_pitch, native_pitch, labels)

    print("\nComplete Processing!")

if __name__ == "__main__":
    main()
