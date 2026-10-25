
import torch

from preprocess import AudioPreprocessor
from acoustic_model import PolyphonicAcousticModel


def main():
    # Module 1: Convert real piano audio into a Mel-spectrogram
    preprocessor = AudioPreprocessor()
    mel_spec = preprocessor.process("test_piano_audio.wav")

    print("Module 1 output:", mel_spec.shape)

    # Convert [1, 229, Time] into [Batch, 1, Time, 229]
    model_input = mel_spec.permute(0, 2, 1).unsqueeze(0)

    print("Module 2 input:", model_input.shape)

    # Module 2: Initialize the acoustic model
    model = PolyphonicAcousticModel()
    model.eval()

    # Generate predictions
    with torch.no_grad():
        onsets, frames = model(model_input)

    print("Onset output:", onsets.shape)
    print("Frame output:", frames.shape)

    print("Onset score range:",
          onsets.min().item(), "to", onsets.max().item())

    print("Frame score range:",
          frames.min().item(), "to", frames.max().item())


if __name__ == "__main__":
    main()
