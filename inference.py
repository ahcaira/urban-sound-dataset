import torch
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
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
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feed_forward_net.pt")
    feed_forward_net.load_state_dict(state_dict)

    _, validation_data = download_mnist_datasets()

    input, target = validation_data[0][0], validation_data[0][1]

    predicted, expected = predict(feed_forward_net, input, target, class_mapping)
    print(f'Predicted: {predicted}, Expected: {expected}')
