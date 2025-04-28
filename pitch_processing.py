import os
import subprocess
import traceback
import librosa
import random
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
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
        import traceback
        traceback.print_exc()
        return None

def visualize_pitch(audio_file, output_dir=None):
    """ 피치 곡선 시각화 (Hz 단위) """
    try:
        # 오디오 로드
        y, sr = librosa.load(audio_file, sr=None)
        
        # 피치 추출 파라미터
        frame_length = 2048
        hop_length = 512
        fmin = librosa.note_to_hz('C2')  # 약 65.4 Hz
        fmax = librosa.note_to_hz('C7')  # 약 2093 Hz
        
        # 피치 추출
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # timestamp 배열 생성
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        
        # 무성음 부분 제거
        pitch_data = f0.copy()
        pitch_data[~voiced_flag] = np.nan
        
        # 피치 곡선 시각화
        plt.figure(figsize=(14, 6), facecolor='white')
        plt.plot(times, pitch_data, color='b', linewidth=2, alpha=0.9)
        plt.title('Pitch Curve (Hz)', fontsize=14)
        plt.xlabel('timestamp (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(times))
        plt.ylim(50, 500) 
        
        plt.tight_layout()
        
        # 결과 저장
        if output_dir:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            plt.savefig(os.path.join(output_dir, f"{base_name}_pitch.png"), dpi=300, bbox_inches='tight', facecolor='white')
        else:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            plt.savefig(f"{base_name}_pitch.png", dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        return True
    except Exception as e:
        print(f"Error: Occurred While Visualizing Pitch Curve: {e}")
        traceback.print_exc()
        return False
    
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

    # 피치 곡선 시각화
    print(f"Visualizing the pitch curve...")
    visualize_pitch(denoised, viz_dir)
    
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