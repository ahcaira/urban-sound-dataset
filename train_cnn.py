import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from urban_sound_dataset import UrbanSoundDataset
from cnn import CNNNet

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
ANNOTATIONS_FILE = 'C:/Users/bemyp/PycharmProjects/add-laugh-tracks/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'
AUDIO_DIR = 'C:/Users/bemyp/PycharmProjects/add-laugh-tracks/data/UrbanSound8K/UrbanSound8K/audio'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f'Loss: {loss.item()}')



def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i + 1}')
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print('-------------------')
    print('Training finished')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    train_data_loader =  DataLoader(usd, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device} device')

    cnn = CNNNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), 'cnn.pth')
    print('Finished Training')

