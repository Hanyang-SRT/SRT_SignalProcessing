import os
import whisper
import json
import torch
import subprocess
import traceback
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate, interp1d
from scipy.stats import zscore
from glob import glob
from fastdtw import fastdtw
from matplotlib.font_manager import FontProperties
from g2pk import G2p

# G2P 객체(한국어 문장을 발음 기호로 변환)
g2p = G2p()

# mp4파일을 wav로 변환
class AudioPreprocessor:
    def __init__(self, base_dir="mp4_clip", sample_rate=16000):
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        # 콘텐츠 목록
        self.available_contents = ['misaeng', 'our_beloved_summer', 'cheese_in_the_trap', 'stove_league']
    
    # 콘텐츠 종류 선택
    def select_contents_directory(self):
        print("\n=== Select Content ===")
        for idx, content in enumerate(self.available_contents, 1):
            print(f"{idx}. {content}")

        while True:
            try:
                choice = input("Enter number (1-4): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.available_contents):
                    return self.available_contents[choice_idx]
                else:
                    print("Please enter a valid number.")
            except ValueError:
                print("Please enter a valid number or drama name.")

    # 저장되어 있는 mp4 파일 중 하나 선택
    def select_file(self, contents_dir, user_filename=None):
        content_path = os.path.join(self.base_dir, contents_dir)
        if not os.path.exists(content_path):
            print(f"Error: Directory not found: {content_path}")
            return None

        mp4_files = glob(os.path.join(content_path, "*.mp4"))
        if not mp4_files:
            print(f"No .mp4 files found in '{content_path}'")
            return None

        if user_filename:
            if not user_filename.lower().endswith(".mp4"):
                user_filename += ".mp4"
            matching_files = [f for f in mp4_files if os.path.basename(f) == user_filename]
            if matching_files:
                print(f"Selected file: {os.path.basename(matching_files[0])}")
                return matching_files[0]
            else:
                print(f"File '{user_filename}' not found.")
                return None
        else:
            print("\nAvailable files:")
            for idx, file in enumerate(mp4_files, 1):
                print(f"{idx}. {os.path.basename(file)}")
            choice = int(input("Select file number: ").strip()) - 1
            if 0 <= choice < len(mp4_files):
                return mp4_files[choice]
            else:
                print("Invalid selection.")
                return None

    # 선택된 mp4 파일을 wav로 변환
    def convert_to_wav(self, input_file, output_file=None):
        try:
            if output_file is None:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_file = f"{base_name}.wav"
            if os.path.abspath(input_file) == os.path.abspath(output_file):
                print("Same input/output file. Skipping conversion.")
                return input_file
            command = ["ffmpeg", "-i", input_file, "-ar", str(self.sample_rate), "-ac", "1", "-y", output_file]
            subprocess.run(command, check=True)
            print(f"WAV file saved: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error during conversion: {e}")
            return None

# 운율 추출 클래스
class PhonemeAnalyzer:
    def __init__(self, whisper_model_size="base"):
        self.model = whisper.load_model(whisper_model_size) # openai 로컬 whisper 로드
    
    # 노이즈 제거
    def spectral_subtraction(self, audio_file, output_file, alpha=2.0, beta=0.05):
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

            enhanced_y = enhanced_y[:len(y)] if len(enhanced_y) > len(y) else np.pad(enhanced_y, (0, len(y) - len(enhanced_y)))
            sf.write(output_file, enhanced_y, sr)
            return output_file
        except Exception as e:
            print(f"[Error] Spectral subtraction: {e}")
            return None
    
    # pitch 추출을 위한 음소 단위 분해
    def extract_phonemes(self, audio_path):
        result = self.model.transcribe(audio_path, language="ko", word_timestamps=True)
        phonemes = []
        for segment in result["segments"]:
            for word_info in segment.get("words", []):
                word = word_info["word"].strip()
                start, end = word_info["start"], word_info["end"]
                duration = end - start
                step = duration / max(1, len(word))
                for i, c in enumerate(word):
                    phonemes.append({"text": c, "timestamp": [start + i * step, start + (i + 1) * step]})
        return phonemes

    # 음소 단위 pitch 추출
    def extract_pitch_sequence(self, audio_path, phonemes):
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
    
    # 발음 기반 음성 보간 : 두 음성 데이터(사용자, native)의 길이가 다를 때 보간
    def interpolate_pitch(self, pitches):
        x = np.arange(len(pitches))
        pitches = np.array(pitches)
        valid = ~np.isnan(pitches)

        if np.sum(valid) < 2:
            return pitches

        f_interp = interp1d(x[valid], pitches[valid], kind='linear', fill_value='extrapolate')
        return f_interp(x)

    def align_pitch_by_phoneme(self, native_pitch, native_labels, user_pitch, user_labels):
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
    
    # 단어 단위 timestamps 추출
    def transcribe_with_word_timestamps(self, audio_path):
        result = self.model.transcribe(audio_path, language="ko", word_timestamps=True)
        
        duration = librosa.get_duration(path=audio_path)
        
        word_chunks = []
        for segment in result["segments"]:
            for word_info in segment.get("words", []):
                word = word_info["word"].strip()
                start, end = word_info["start"], word_info["end"]
                if end is None:
                    end = duration
                
                word_chunks.append({
                    "text": word,
                    "timestamp": (round(start, 2), round(end, 2))
                })
        
        return word_chunks

    # 단어별 intensity 추출
    def get_intensity_per_chunk(self, audio_file, chunks):
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

    
    # intensity 비교 분석
    def compare_intensity(self, words_ref, intensities_ref, words_usr, intensities_usr):
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

        return usr_z.tolist(), ref_z.tolist(), avg_score, highlights, feedbacks

    # duration 추출
    def get_duration_per_chunk(self, chunks, audio_file, min_duration=0.05, max_duration=2.0, silence_threshold=0.01):
        durations = []
        valid_chunks = []
        y, sr = librosa.load(audio_file, sr=None)
        total_audio_duration = len(y) / sr

        for chunk in chunks:
            start_time, end_time = chunk['timestamp']
            if end_time is None:
                end_time = total_audio_duration
                chunk['timestamp'] = (start_time, end_time)

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]

            # 앞부분 무음 제거
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]

            non_silent_frames = np.where(rms > silence_threshold)[0]
            if len(non_silent_frames) > 0:
                first_voice_frame = non_silent_frames[0]
                start_offset = first_voice_frame * hop_length
                segment = segment[start_offset:]
                start_time += start_offset / sr

            duration = end_time - start_time
            if duration < min_duration or duration > max_duration:
                continue

            durations.append(duration)
            chunk['timestamp'] = (start_time, end_time)
            valid_chunks.append(chunk)
            
        return valid_chunks, durations

    # 상대 duration 계산
    def get_relative_durations(self, durations):
        total = sum(durations)
        return [d / total for d in durations] if total > 0 else [0] * len(durations)

    # DTW를 사용한 duration 분석
    def analyze_duration_with_dtw(self, user_chunks, ref_chunks, user_durations, ref_durations):
        if not user_durations or not ref_durations:
            return [], 0, []

        user_rel = np.array(self.get_relative_durations(user_durations))[:, None]
        ref_rel = np.array(self.get_relative_durations(ref_durations))[:, None]

        _, path = fastdtw(user_rel, ref_rel)
        abs_diffs = [abs(user_rel[i][0] - ref_rel[j][0]) for i, j in path]
        mae = np.mean(abs_diffs)
        similarity_score = round(max(0, 100 * (1 - mae)))

        feedbacks, highlights = [], []
        for idx, (i, j) in enumerate(path):
            if i >= len(user_chunks): break
            word = user_chunks[i]['text']
            diff = user_rel[i][0] - ref_rel[j][0]
            highlights.append(abs(diff) > 0.1)
            if diff > 0.1:
                feedbacks.append(f"'{word}' 단어를 상대적으로 길게 발음했습니다.")
            elif diff < -0.1:
                feedbacks.append(f"'{word}' 단어를 상대적으로 짧게 발음했습니다.")
        if not feedbacks:
            feedbacks.append("모든 단어의 발화 길이가 적절했습니다.")
        return feedbacks, similarity_score, highlights

    # 결과 시각화
    def result_visualize(self, pitch_data, duration_data, stress_data, labels, user_words, output_dir="visualization"):
        os.makedirs(output_dir, exist_ok=True)

        # 데이터 길이 맞춤 함수
        def align_data_lengths(data1, data2, target_labels):
            min_len = min(len(data1), len(data2), len(target_labels))
            return data1[:min_len], data2[:min_len], target_labels[:min_len]
        
        plt.ion()
        # Pitch Plot (음소 단위)
        if pitch_data['user'] and pitch_data['native'] and labels:
            user_pitch, native_pitch, pitch_labels = align_data_lengths(
                pitch_data['user'], pitch_data['native'], labels
            )
            
            fig1 = plt.figure(figsize=(10, 5), facecolor="#A49CC7")
            x = range(len(pitch_labels))
            plt.plot(x, native_pitch, label="Native", color="skyblue", linewidth=2)
            plt.plot(x, user_pitch, label="User", color="indigo", linewidth=2)
            plt.scatter(x, native_pitch, color='skyblue', edgecolors='skyblue', zorder=5)
            plt.scatter(x, user_pitch, color='indigo', edgecolors='indigo', zorder=5)
            plt.xticks(x, pitch_labels, fontproperties=FontProperties(family='AppleGothic'))
            plt.yticks(color='white')
            plt.title(f"Pitch Analysis", fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # 그래프 파일 저장
            plt.savefig(os.path.join(output_dir, "pitch.png"), dpi=150, bbox_inches='tight')
            plt.draw() 
            plt.pause(0.1)

        # intensity, duration 피드백 
        if user_words: 
            # 사용자가 발음한 문장을 단어 단위로 재구성
            user_sentence = " ".join([word["text"] for word in user_words])
            fig2, ax = plt.subplots(figsize=(10, 5))
 
            fig2.patch.set_facecolor('#8B7FA8')
            ax.set_facecolor('#A49CC7')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')

            ax.text(5, 9, 'Intensity, Duration Analysis', fontsize=24, fontweight='bold', 
                    ha='center', va='center', color='white',
                    fontproperties=FontProperties(family='AppleGothic'))

            sentence_box = plt.Rectangle((1, 6.5), 8, 1.5, 
                                    facecolor='#5D4E75', edgecolor='#4A3D5C', 
                                    linewidth=2, alpha=0.9)
            ax.add_patch(sentence_box)
            
            ax.text(5, 7.25, user_sentence, fontsize=18, fontweight='bold',
                    ha='center', va='center', color='white',
                    fontproperties=FontProperties(family='AppleGothic'), wrap=True)
            
            # 강세 분석 제목 - 박스 밖으로 배치하여 명확한 구조 생성
            ax.text(1.5, 5.8, 'Intensity:', fontsize=14, fontweight='bold',
                    ha='left', va='center', color='white',
                    fontproperties=FontProperties(family='AppleGothic'))
            
            # Stress 피드백
            intensity_box = plt.Rectangle((1, 4.5), 8, 1.2, 
                                    facecolor='#6B5B73', edgecolor='#5D4E75', 
                                    linewidth=1, alpha=0.8)
            ax.add_patch(intensity_box)

            intensity_feedback = stress_data.get('feedback', '강세 분석 완료')
            if isinstance(intensity_feedback, list) and len(intensity_feedback) > 0:
                intensity_feedback = intensity_feedback[0]  
            
            ax.text(5, 5.1, intensity_feedback, fontsize=12,
                    ha='center', va='center', color='white',
                    fontproperties=FontProperties(family='AppleGothic'), wrap=True)
            
            ax.text(1.5, 3.8, 'Duration:', fontsize=14, fontweight='bold',
                    ha='left', va='center', color='white', 
                    fontproperties=FontProperties(family='AppleGothic'))
            
            # Duration 피드백
            duration_box = plt.Rectangle((1, 2.5), 8, 1.2, 
                                    facecolor='#6B5B73', edgecolor='#5D4E75', 
                                    linewidth=1, alpha=0.8)
            ax.add_patch(duration_box)
            
            duration_feedback = duration_data.get('feedback', '발화 길이 분석 완료')
            if isinstance(duration_feedback, list) and len(duration_feedback) > 0:
                duration_feedback = duration_feedback[0]
                
            # 발화 길이 분석 내용
            ax.text(5, 3.1, duration_feedback, fontsize=12,
                    ha='center', va='center', color='white',
                    fontproperties=FontProperties(family='AppleGothic'), wrap=True)
            
            plt.tight_layout()
            
            # 피드백 리포트 파일 저장
            plt.savefig(os.path.join(output_dir, "feedback_report.png"), 
                    dpi=150, bbox_inches='tight', facecolor='#8B7FA8')
            plt.draw()
            plt.pause(0.1)
            
        #     print(f"피드백 리포트가 '{output_dir}/feedback_report.png'에 저장되었습니다.")

        # # 사용자 경험 개선을 위한 종료 처리
        # print(f"\n시각화 완료! 그래프들이 '{output_dir}' 폴더에 저장되었습니다.")

        try:
            input()  
        except KeyboardInterrupt:
            pass 
        finally:
            plt.ioff()  
            plt.close('all')  

def main():
    preprocessor = AudioPreprocessor()
    analyzer = PhonemeAnalyzer()

    # 콘텐츠 디렉토리 및 파일 선택 
    drama_dir = preprocessor.select_contents_directory()
    drama_filename = input("\nEnter the drama file name to process (without extension): ").strip()
    drama_file = preprocessor.select_file(drama_dir, drama_filename)
    if not drama_file:
        print("The file to process could not be found. The program will terminate.")
        return

    # 사용자 음성 파일 입력 
    user_input = input("User voice file(wav): ").strip()
    if not os.path.exists(user_input):
        print("User voice file not found.")
        return

    # # 필요한 디렉토리 생성 - 중간 처리 파일들을 저장할 폴더들을 미리 생성
    # os.makedirs("wav_file", exist_ok=True)
    # os.makedirs("denoised_audio", exist_ok=True)

    #  wav변환 및 노이즈 제거
    base_native = os.path.splitext(os.path.basename(drama_file))[0]
    native_wav = preprocessor.convert_to_wav(drama_file, f"wav_file/{base_native}.wav")
    native_clean = analyzer.spectral_subtraction(native_wav, f"denoised_audio/{base_native}_denoised.wav")
    user_clean = analyzer.spectral_subtraction(user_input, f"denoised_audio/user_denoised.wav")

    # 음소 추출 (pitch)
    native_phonemes = analyzer.extract_phonemes(native_clean)
    user_phonemes = analyzer.extract_phonemes(user_clean)

    print("\nNative phonemes:", [p["text"] for p in native_phonemes])
    print("User phonemes:", [p["text"] for p in user_phonemes])

    # 단어 추출 (intensity/duration)
    native_words = analyzer.transcribe_with_word_timestamps(native_clean)
    user_words = analyzer.transcribe_with_word_timestamps(user_clean)

    print("\nNative words:", [w["text"] for w in native_words])
    print("User words:", [w["text"] for w in user_words])

    # 운율 분석
    native_pitch, native_labels = analyzer.extract_pitch_sequence(native_clean, native_phonemes)
    user_pitch, user_labels = analyzer.extract_pitch_sequence(user_clean, user_phonemes)
    native_pitch = analyzer.interpolate_pitch(native_pitch)
    user_pitch = analyzer.interpolate_pitch(user_pitch)
    user_pitch_aligned, native_pitch_aligned, labels = analyzer.align_pitch_by_phoneme(
        native_pitch, native_labels, user_pitch, user_labels
    )
    
    # Pitch 점수 계산
    if len(user_pitch_aligned) > 0 and len(native_pitch_aligned) > 0:
        pitch_diff = np.array(user_pitch_aligned) - np.array(native_pitch_aligned)
        pitch_score = max(0, int(100 - np.nanmean(np.abs(pitch_diff)) / 10))
    else:
        pitch_score = 0

    words_usr, intensities_usr = analyzer.get_intensity_per_chunk(user_clean, user_words)
    words_ref, intensities_ref = analyzer.get_intensity_per_chunk(native_clean, native_words)
    usr_z, ref_z, stress_score, stress_highlights, intensity_feedbacks = analyzer.compare_intensity(
        words_ref, intensities_ref, words_usr, intensities_usr
    )

    user_chunks_filt, user_durations = analyzer.get_duration_per_chunk(user_words, user_clean)
    ref_chunks_filt, ref_durations = analyzer.get_duration_per_chunk(native_words, native_clean)
    duration_feedbacks, duration_score, duration_highlights = analyzer.analyze_duration_with_dtw(
        user_chunks_filt, ref_chunks_filt, user_durations, ref_durations
    )

    # 시각화
    pitch_data = {
        "user": user_pitch_aligned,
        "native": native_pitch_aligned,
        "score": pitch_score
    }
    
    duration_data = {
        "user": user_durations,
        "native": ref_durations,
        "score": duration_score,
        "feedback": duration_feedbacks[0] if duration_feedbacks else "피드백 없음"
    }
    
    stress_data = {
        "user": usr_z,
        "native": ref_z,
        "score": stress_score,
        "feedback": intensity_feedbacks[0] if intensity_feedbacks else "피드백 없음"
    }

    analyzer.result_visualize(pitch_data, duration_data, stress_data, labels, user_words)
    
    # JSON 결과 반환
    result = {
        "syllables": labels,
        
        # Pitch 분석 결과
        "pitch": {
            "user": [round(float(p), 2) if not np.isnan(p) else None for p in user_pitch_aligned],
            "native": [round(float(p), 2) if not np.isnan(p) else None for p in native_pitch_aligned],
            "score": int(pitch_score)
        },
        
        # Intensity 분석 결과
        "intensity": {
            "user": [round(float(z), 2) for z in usr_z],
            "native": [round(float(z), 2) for z in ref_z],
            "score": int(stress_score),
            "highlight": [bool(h) for h in stress_highlights],
            "feedback": list(intensity_feedbacks)
        },
        
        # Duration 분석 결과
        "duration": {
            "user": [round(float(d), 2) for d in user_durations],
            "native": [round(float(d), 2) for d in ref_durations],
            "score": int(duration_score),
            "highlight": [bool(h) for h in duration_highlights[:len(user_durations)]] if duration_highlights else [],
            "feedback": list(duration_feedbacks)
        }
    }
    
    # JSON 결과 출력 - 구조화된 분석 결과를 표준 형식으로 표시
    print("\n=== Analysis Results (JSON Format) ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
