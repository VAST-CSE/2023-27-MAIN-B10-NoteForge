import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf  # Import soundfile directly


class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=229, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=self.n_mels
        )

        self.amp_to_db = T.AmplitudeToDB(stype='power', top_db=80)

    def load_and_standardize(self, file_path):
        # 1. Read audio directly with soundfile (bypasses torchaudio/ffmpeg completely)
        wav_data, sr = sf.read(file_path)

        # 2. Convert to PyTorch tensor
        # soundfile loads as numpy array [Time, Channels]. We need [Channels, Time]
        if wav_data.ndim > 1:
            waveform = torch.tensor(wav_data).float().t()  # Transpose to [Channels, Time]
            # Average channels to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            # Mono case
            waveform = torch.tensor(wav_data).float().unsqueeze(0)

        # 3. Resample to target sample rate (16kHz)
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def process(self, file_path):
        waveform = self.load_and_standardize(file_path)
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amp_to_db(mel_spec)
        return log_mel_spec

if __name__ == "__main__":

    # Example Usage
    preprocessor = AudioPreprocessor()
    input_tensor = preprocessor.process("test_piano_audio.wav")
    print(f"Output Tensor Shape: {input_tensor.shape}")

    # Convert spectrogram to a 2D array
    spectrogram_2d = input_tensor.squeeze(0).numpy()

    # Plot the spectrogram
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.imshow(
        spectrogram_2d,
        aspect="auto",
        origin="lower",
        cmap="magma"
    )

    plt.colorbar(format="%+2.0f dB", label="Loudness (Decibels)")
    plt.title("Mel-Spectrogram Visualization")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.tight_layout()
    plt.show()
