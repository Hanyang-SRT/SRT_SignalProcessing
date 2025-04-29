from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

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
    for chunk in result['chunks']:
        print(chunk['text'], chunk['timestamp'])
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

        if len(segment) == 0:
            intensities.append(0.0)
        else:
            rms = np.sqrt(np.mean(segment**2))
            intensities.append(rms)

        words.append(chunk['text'])

    return words, intensities

def plot_intensity_toplines(chunks, intensities, scale_factor=10):
   
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 4))

    scaled_intensities = [i * scale_factor for i in intensities]

    for chunk, intensity in zip(chunks, scaled_intensities):
        if chunk['timestamp'] is None or chunk['timestamp'][0] is None or chunk['timestamp'][1] is None:
            continue
        
        start_time, end_time = chunk['timestamp']
        duration = end_time - start_time

        ax.plot([start_time, end_time], [intensity, intensity], color='blue', linewidth=2)

        # 단어 표시
        mid = (start_time + end_time) / 2
        text_y = intensity + (max(scaled_intensities) * 0.05)
        ax.text(mid, text_y, chunk['text'], ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(scaled_intensities) * 1.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Intensity (scaled x{scale_factor})')
    ax.set_title('Word-wise Intensity (Top Line Only)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



input_audio = "note.wav"             
denoised_audio = "note_denoised.wav"

clean_audio = spectral_subtraction(input_audio, denoised_audio)

if clean_audio:
    result = transcribe_with_whisper(clean_audio)
    chunks = result['chunks']
    chunks = sorted(chunks, key=lambda x: (x['timestamp'] if x['timestamp'] else [float('inf'), float('inf')]))
    words, intensities = get_intensity_per_chunk(clean_audio, chunks)
    
    plot_intensity_toplines(chunks, intensities)

