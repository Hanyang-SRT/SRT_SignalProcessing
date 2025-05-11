import os
import whisper
import torch
import subprocess
import traceback
import librosa
import random
import matplotlib
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import interpolate 
from glob import glob

def convert_to_wav(input_file, output_file=None, sample_rate=16000):
    """오디오 파일을 WAV 형식으로 변환"""
    try:
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{base_name}.wav"
        
        command = [
            "ffmpeg", "-i", input_file, 
            "-ar", str(sample_rate), 
            "-ac", "1",  
            "-y",  
            output_file
        ]

        subprocess.run(command, check=True)
        print(f"Conversion Complete: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error: Occurred During Conversion {e}")
        return None

def spectral_subtraction(audio_file, output_file, alpha=2.0, beta=0.05):
    """ 스펙트럴 서브트랙션을 적용하여 노이즈 제거 """
    try:
        # 오디오 로드
        y, sr = librosa.load(audio_file, sr=None)
        
        # STFT 적용
        frame_length = 1024
        hop_length = 512
        
        D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, win_length=frame_length)
        magnitude, phase = np.abs(D), np.angle(D)
        power = magnitude**2
        
        # 노이즈 추정
        noise_frames = min(int(0.5 * sr / hop_length), 10)
        noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
        
        # 스펙트럴 서브트랙션 이득 계산
        gain = np.maximum(1 - alpha * (noise_power / (power + 1e-10)), beta)
        
        # 향상된 스펙트럼 계산
        enhanced_magnitude = magnitude * gain
        enhanced_D = enhanced_magnitude * np.exp(1j * phase)
        
        # 역변환
        enhanced_y = librosa.istft(enhanced_D, hop_length=hop_length, win_length=frame_length)
        
        # 길이 맞추기
        if len(enhanced_y) > len(y):
            enhanced_y = enhanced_y[:len(y)]
        elif len(enhanced_y) < len(y):
            enhanced_y = np.pad(enhanced_y, (0, len(y) - len(enhanced_y)))
            
        # 결과 저장
        sf.write(output_file, enhanced_y, sr)
        print(f"Save audio with noise removed: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"Error: Applying Spectral Subtraction: {e}")
        traceback.print_exc()
        return None

def extract_timestamps(audio_file):
    """Whisper를 사용하여 오디오 파일에서 단어 단위 타임스탬프 추출"""
    try:
        import whisper
        import torch
        
        print("Whisper 모델 로딩 중...")
        # 기본적으로 'base' 모델 사용 (더 정확한 결과를 원하면 'small', 'medium', 'large' 사용 가능)
        model = whisper.load_model("large")
        
        # 한국어 인식 명시적 설정
        print(f"{audio_file} 음성 인식 중...")
        result = model.transcribe(audio_file, language="ko", word_timestamps=True)
        
        # 단어 단위 타임스탬프를 담을 리스트 초기화
        word_chunks = []
        
        # 세그먼트별로 단어 추출
        for segment in result["segments"]:
            # 각 세그먼트 내의 단어별 정보 추출
            if "words" in segment:
                for word_info in segment["words"]:
                    word_text = word_info["word"].strip()
                    word_start = word_info["start"]
                    word_end = word_info["end"]
                    
                    # 단어 저장 (공백 제거)
                    if word_text and not word_text.isspace():
                        word_chunks.append({
                            'text': word_text,
                            'timestamp': [word_start, word_end]
                        })
        
        print(f"음성 인식 완료: {len(word_chunks)}개 단어 추출됨")
        
        # 음소/글자 단위 청크로 분할
        phoneme_chunks = []
        for word_chunk in word_chunks:
            word = word_chunk['text']
            start_time, end_time = word_chunk['timestamp']
            duration = end_time - start_time
            
            # 단어를 개별 글자로 분할
            chars = list(word)
            if len(chars) > 0:  # 빈 단어가 아닌 경우만 처리
                char_duration = duration / len(chars)
                
                for i, char in enumerate(chars):
                    char_start = start_time + (i * char_duration)
                    char_end = start_time + ((i + 1) * char_duration)
                    
                    phoneme_chunks.append({
                        'text': char,
                        'timestamp': [char_start, char_end]
                    })
        
        print(f"음소 단위 분할 완료: {len(phoneme_chunks)}개 음소 추출됨")
        
        return {
            'word_chunks': word_chunks,
            'phoneme_chunks': phoneme_chunks
        }
    except Exception as e:
        print(f"음성 인식 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            'word_chunks': [],
            'phoneme_chunks': []
        }

def visualize_pitch_relative(audio_file, phoneme_chunks=None, output_dir=None):
    """ 음소별 상대적 높낮이 시각화 """
    try:
        # 오디오 로드 및 피치 추출
        y, sr = librosa.load(audio_file, sr=None)
        
        frame_length = 2048
        hop_length = 512
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C7')
        
        # 기본 설정으로 피치 추출
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=hop_length
        )
        
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        
        # 유효한 음소 청크 필터링
        if phoneme_chunks and isinstance(phoneme_chunks, list):
            valid_chunks = [chunk for chunk in phoneme_chunks 
                           if chunk['timestamp'] and chunk['timestamp'][0] is not None 
                           and chunk['timestamp'][1] is not None]
        else:
            valid_chunks = []
        
        # 각 음소별 대표 피치값 계산
        phoneme_pitches = []
        phoneme_positions = []
        
        for chunk in valid_chunks:
            start_time, end_time = chunk['timestamp']
            
            # 해당 시간 범위의 피치 데이터 찾기
            start_idx = np.argmin(np.abs(times - start_time))
            end_idx = np.argmin(np.abs(times - end_time))
            
            segment_pitch = f0[start_idx:end_idx+1]
            
            # 유효한 피치값들 필터링 (NaN이 아닌 값들)
            valid_segment = segment_pitch[~np.isnan(segment_pitch)]
            
            if len(valid_segment) > 0:
                # 중앙값 사용 (이상치에 덜 민감)
                pitch_value = np.median(valid_segment)
            else:
                # 피치를 찾을 수 없는 경우 None으로 설정
                pitch_value = None
            
            phoneme_pitches.append(pitch_value)
            phoneme_positions.append((start_time + end_time) / 2)
        
        # None 값들을 선형 보간으로 채우기
        valid_indices = [i for i, val in enumerate(phoneme_pitches) if val is not None]
        
        if len(valid_indices) >= 2:
            # 유효한 값들로 선형 보간 함수 생성
            interp_func = interpolate.interp1d(
                [phoneme_positions[i] for i in valid_indices],
                [phoneme_pitches[i] for i in valid_indices],
                kind='linear', fill_value='extrapolate'
            )
            
            # None 값들을 보간된 값으로 채우기
            for i, val in enumerate(phoneme_pitches):
                if val is None:
                    phoneme_pitches[i] = interp_func(phoneme_positions[i])
        
        # 모든 값이 None인 경우 기본값 설정
        elif not any(val is not None for val in phoneme_pitches):
            # 기본 피치 범위 설정
            default_pitch = 150.0  # 예시 값
            phoneme_pitches = [default_pitch] * len(phoneme_chunks)
        
        # 시각화
        plt.figure(figsize=(14, 8), facecolor='white')
        
        # 유효한 피치값이 있는 경우에만 그래프 그리기
        if phoneme_pitches and all(p is not None for p in phoneme_pitches):
            # 전체 피치 범위 계산
            min_pitch = min(phoneme_pitches)
            max_pitch = max(phoneme_pitches)
            pitch_range = max_pitch - min_pitch if min_pitch != max_pitch else 20
            
            # 음소 간 연결선 그리기
            plt.plot(phoneme_positions, phoneme_pitches, 'b-', linewidth=2, alpha=0.8)
            
            # 각 음소 지점에 점 표시
            plt.scatter(phoneme_positions, phoneme_pitches, c='red', s=50, zorder=5)
            
            # 음소별 텍스트 표시
            for i, chunk in enumerate(valid_chunks):
                if i < len(phoneme_positions) and i < len(phoneme_pitches):
                    plt.text(phoneme_positions[i], phoneme_pitches[i] + pitch_range*0.02, 
                            chunk['text'], ha='center', va='bottom', fontsize=20, color='black',
                            fontproperties=matplotlib.font_manager.FontProperties(family='AppleGothic'),
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            # 그래프 설정
            padding = pitch_range * 0.2
            plt.ylim(min_pitch - padding, max_pitch + padding)
        
        plt.title('Relative Pitch by Phoneme', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Relative Pitch (Hz)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 결과 저장
        if output_dir:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            plt.savefig(os.path.join(output_dir, f"{base_name}_relative_pitch.png"), 
                       dpi=300, bbox_inches='tight', facecolor='white')
        else:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            plt.savefig(f"{base_name}_relative_pitch.png", 
                      dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        return True
    except Exception as e:
        print(f"Error visualizing relative pitch: {e}")
        traceback.print_exc()
        return False

# def visualize_pitch(audio_file, phoneme_chunks=None, output_dir=None):
#     """ 하나의 플롯에서 인식된 음소 부분에 초점을 맞춘 피치 곡선 시각화 """
#     try:
#         # 오디오 로드 및 피치 추출 (기존 코드와 동일)
#         y, sr = librosa.load(audio_file, sr=None)
        
#         frame_length = 2048
#         hop_length = 512
#         fmin = librosa.note_to_hz('C2')
#         fmax = librosa.note_to_hz('C7')
        
#         f0, voiced_flag, voiced_probs = librosa.pyin(
#             y, fmin=fmin, fmax=fmax, sr=sr,
#             frame_length=frame_length, hop_length=hop_length
#         )
        
#         times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
#         pitch_data = f0.copy()
#         pitch_data[~voiced_flag] = np.nan
        
#         # 유효한 음소 청크 필터링
#         if phoneme_chunks and isinstance(phoneme_chunks, list):
#             valid_chunks = [chunk for chunk in phoneme_chunks 
#                            if chunk['timestamp'] and chunk['timestamp'][0] is not None 
#                            and chunk['timestamp'][1] is not None]
#         else:
#             valid_chunks = []
        
#         # 시각화
#         plt.figure(figsize=(14, 8), facecolor='white')
        
#         # 전체 피치 곡선 (회색으로 흐리게 표시)
#         plt.plot(times, pitch_data, color='lightgray', linewidth=1, alpha=0.5)
        
#         # 피치 데이터의 범위 계산
#         valid_pitch = pitch_data[~np.isnan(pitch_data)]
#         min_pitch = np.min(valid_pitch) if len(valid_pitch) > 0 else 100
#         max_pitch = np.max(valid_pitch) if len(valid_pitch) > 0 else 400
#         pitch_range = max_pitch - min_pitch
        
#         # 텍스트 위치 조정을 위한 변수
#         text_positions = []  # (x, y, width, height) 형태로 텍스트 위치 저장
        
#         # 음소 발화 부분 강조 및 텍스트 배치
#         for chunk in valid_chunks:
#             start_time, end_time = chunk['timestamp']
#             text = chunk['text']
            
#             # 해당 시간 범위의 피치 데이터 찾기
#             start_idx = np.argmin(np.abs(times - start_time))
#             end_idx = np.argmin(np.abs(times - end_time))
            
#             segment_times = times[start_idx:end_idx+1]
#             segment_pitch = pitch_data[start_idx:end_idx+1]
            
#             # 해당 구간 피치 데이터 그리기
#             # plt.plot(segment_times, segment_pitch, color='blue', linewidth=2)
            
#             # 해당 구간 평균 피치 계산 (텍스트 배치용)
#             valid_segment = segment_pitch[~np.isnan(segment_pitch)]
#             if len(valid_segment) > 0:
#                 avg_pitch = np.mean(valid_segment)
#             else:
#                 avg_pitch = min_pitch + pitch_range * 0.3  # 기본값
            
#             # 음영 처리
#             plt.axvspan(start_time, end_time, alpha=0.1, color='orange')
            
#             # 텍스트 위치 계산 (피치 곡선 바로 위)
#             text_offset = pitch_range * 0.1  # 기본 오프셋
#             text_x = (start_time + end_time) / 2
            
#             # 겹침 방지를 위한 텍스트 위치 조정
#             text_y = avg_pitch + text_offset
#             text_width = (end_time - start_time) * 0.8  # 텍스트 가로 길이 추정
#             text_height = pitch_range * 0.08  # 텍스트 세로 길이 추정
            
#             # 기존 텍스트들과 겹치는지 확인하고 위치 조정
#             overlap = True
#             attempts = 0
#             while overlap and attempts < 10:
#                 overlap = False
#                 for pos in text_positions:
#                     px, py, pw, ph = pos
#                     # 텍스트 박스 겹침 검사 (간단한 충돌 감지)
#                     if (abs(text_x - px) < (text_width + pw) / 2 and 
#                         abs(text_y - py) < (text_height + ph) / 2):
#                         overlap = True
#                         text_y += text_height * 1.2  # 겹침 발생 시 위로 이동
#                         break
#                 attempts += 1
            
#             # 텍스트 표시
#             plt.text(text_x, text_y, text, 
#                     ha='center', va='bottom', fontsize=10, color='red',
#                     fontproperties=matplotlib.font_manager.FontProperties(family='AppleGothic'),
#                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
#             # 텍스트 위치 기록
#             text_positions.append((text_x, text_y, text_width, text_height))
            
#             # 텍스트와 피치 곡선 연결선 (선택 사항)
#             # plt.plot([text_x, text_x], [avg_pitch, text_y - text_height/2], 
#             #         color='gray', linestyle='--', alpha=0.5)
        
#         plt.title('Pitch Curve with Dynamic Text Placement (Hz)', fontsize=14)
#         plt.xlabel('Time (s)', fontsize=12)
#         plt.ylabel('Frequency (Hz)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.xlim(0, max(times))
        
#         # 충분한 여유 공간 확보
#         padding = pitch_range * 0.5
#         plt.ylim(min_pitch - padding * 0.2, max_pitch + padding)
        
#         plt.tight_layout()
        
#         # 결과 저장
#         if output_dir:
#             base_name = os.path.splitext(os.path.basename(audio_file))[0]
#             plt.savefig(os.path.join(output_dir, f"{base_name}_dynamic_text.png"), 
#                        dpi=300, bbox_inches='tight', facecolor='white')
#         else:
#             base_name = os.path.splitext(os.path.basename(audio_file))[0]
#             plt.savefig(f"{base_name}_dynamic_text.png", 
#                       dpi=300, bbox_inches='tight', facecolor='white')
        
#         plt.show()
        
#         return True
#     except Exception as e:
#         print(f"Error visualizing pitch with dynamic text: {e}")
#         traceback.print_exc()
#         return False
    
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

def process_single_mp4_file(mp4_file):
    """ 단일 MP4 파일 처리 """
    if not mp4_file or not os.path.exists(mp4_file):
        print("This is not a valid MP4 file.")
        return False
    
    base_name = os.path.splitext(os.path.basename(mp4_file))[0]
    
    # 출력 디렉토리 설정
    wav_dir = os.path.join(".", "wav_file")
    denoise_dir = os.path.join(".", "denoised_audio")
    viz_dir = os.path.join(".", "visualization")
    
    # 디렉토리가 없으면 생성
    for directory in [wav_dir, denoise_dir, viz_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # WAV 파일로 변환 
    wav_file = os.path.join(wav_dir, f"{base_name}.wav")
    converted_file = convert_to_wav(mp4_file, wav_file)
    
    if not converted_file:
        print(f"{mp4_file} Conversion Failed.")
        return False
    
    # 노이즈 제거 
    denoised_file = os.path.join(denoise_dir, f"{base_name}_denoised.wav")
    denoised = spectral_subtraction(converted_file, denoised_file)
    
    if not denoised:
        print(f"{converted_file} 노이즈 제거 실패.")
        return False

    # 음성 인식 및 음소 타임스탬프 추출
    print("Extracting speech timestamps...")
    timestamps = extract_timestamps(denoised)
    phoneme_chunks = timestamps['phoneme_chunks']

    # 피치 곡선 시각화 (음소 타임라인 포함)
    print(f"Visualizing the pitch curve with phonemes...")
    # visualize_pitch(denoised, phoneme_chunks, viz_dir)
    visualize_pitch_relative(denoised, phoneme_chunks, viz_dir)
    return True

def main(): 
    # 컨텐츠 디렉토리 선택
    drama_dir = select_contents_directory()
    if not drama_dir:
        print("Content selection has been cancelled.")
        return
    
    # 사용자 파일명 입력
    user_filename = input("\nEnter the specific file name to be processed (extension can be omitted): ").strip()
    selected_file = select_file(drama_dir, user_filename)
    
    if not selected_file:
        print("The file to process could not be found. The program will terminate.")
        return
    
    # 파일 처리
    process_single_mp4_file(selected_file)

if __name__ == "__main__":
    main()