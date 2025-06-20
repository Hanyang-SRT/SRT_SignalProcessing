from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
from pitch_analysis import run_pitch_analysis
from intensity_analysis import run_intensity_analysis
from duration_analysis import run_duration_analysis
import logging
import torch
import os

if False:
    ORIGIN = "0.0.0.0"
    PORT = 8000
else:
    ORIGIN = "https://areum817-speech-recognition.hf.space"
    PORT = 80

# MPS 문제 방지
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.backends.mps.is_available = lambda: False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    return_timestamps="word",
    device=0 if torch.cuda.is_available() else -1
)

# Whisper로 텍스트 및 타임스탬프 추출
def transcribe_with_whisper(audio_file):
    # try:
    logger.info(f"Whisper 전사 시작: {audio_file}")
    result = whisper_pipe(audio_file)

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

# 통합 분석 실행 함수
def run_integrated_analysis(user_audio, ref_audio):
    base64_bool = True
    
    logger.info(f"=== 통합 분석 시작 ===")
    logger.info(f"User: {user_audio}, Ref: {ref_audio}")

    user_denoised = spectral_subtraction(user_audio, "user_clean.wav")
    ref_denoised = spectral_subtraction(ref_audio, "ref_clean.wav")
    user_result = transcribe_with_whisper(user_denoised)
    ref_result = transcribe_with_whisper(ref_denoised)
    
    pitch_result = run_pitch_analysis(
        user_denoised, ref_denoised, 
        user_result['chunks'], ref_result['chunks']
    )

    intensity_result = run_intensity_analysis(
        user_denoised, ref_denoised, 
        user_result['chunks'], ref_result['chunks']
    )
    
    duration_result = run_duration_analysis(
        user_denoised, ref_denoised, 
        user_result['chunks'], ref_result['chunks']
    )
    
    integrated_result = {}
    integrated_result.update(pitch_result)
    integrated_result.update(intensity_result)
    integrated_result.update(duration_result)
    
    return integrated_result