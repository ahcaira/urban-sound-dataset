import torch
import torchaudio

from cnn import CNNNet
from urban_sound_dataset import UrbanSoundDataset
from train_cnn import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected



if __name__ == '__main__':
    cnn = CNNNet()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, 'cpu')


    input, target = usd[0][0], usd[0][1]
    input.unsqueeze_(0)

    predicted, expected = predict(cnn, input, target, class_mapping)
    print(f'Predicted: {predicted}, Expected: {expected}')
