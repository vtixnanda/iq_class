import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import nn,save,load
from torch.optim import Adam
from src.iqdataset import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class IQClassifier(nn.Module):
    def __init__(self, slice_size, num_classes):
        super(IQClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=7, padding='same')
        conv2 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        pool = nn.MaxPool1d(kernel_size=2)

        convs = nn.Sequential()
        for _ in range(4):
            convs.append(nn.Conv1d(128, 128, kernel_size=7, padding='same'))
            convs.append(nn.Conv1d(128, 128, kernel_size=5, padding='same'))
            convs.append(nn.MaxPool1d(kernel_size=2))

        flatten = nn.Flatten()
        fc1 = nn.Linear(128 * (slice_size // 32), 256)  # Adjust input size for the fully connected layers
        fc2 = nn.Linear(256, 128)
        fc3 = nn.Linear(128, num_classes)

        self.model = nn.Sequential(
            conv1, conv2, pool, convs, flatten, fc1, fc2, fc3
        )

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, x):
        x = self.model(x)
        return x

    def train(self, epochs, train_loader):
        for epoch in range(epochs):  # Train for 10 epochs
            for iq, labels in train_loader:
                iq, labels = iq.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # Reset gradients
                outputs = self.forward(iq)  # Forward pass

                loss = self.loss_fn(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

            print(f"Epoch:{epoch} loss is {loss.item()}")

if __name__ == '__main__':
    # Load a dataset
    folder = '/home/vn5378/Documents/DR-RFF/datasets/KRI-16IQImbalances-DemodulatedData/'
    file_pre = 'Demod_WiFi_cable_X310_3123D76_IQ#'
    file_post = '_run1.sigmf-'
    types = ['data', 'meta']
    dataset = {i: [file_pre + str(i) + file_post + ty for ty in types] for i in range(1, 33)}
    indices = [i for i in range(1,33)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    remove = []
    for i, files in dataset.items():
        for f in files:
            if not os.path.exists(folder+f):
                remove.append(i)
                break

    for i in remove:
        dataset.pop(i)
        indices.remove(i)

    chunk = 1024
    labels = 16

    classifier = IQClassifier(chunk, labels)
    IQData = IQDataset(dataset, folder, indices)
    generator1 = torch.Generator().manual_seed(42)
    train, val, test = random_split(IQData, [0.7, 0.2, 0.1], generator=generator1)

    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=True)
    # classifier.train(160, train_loader)
    # torch.save(classifier.state_dict(), 'classifier_state.pt')

    # Load the saved model
    with open('classifier_state.pt', 'rb') as f: 
        classifier.load_state_dict(load(f, weights_only=False))

    iq_features, iq_labels = next(iter(test_loader))
    iq_features, iq_labels = iq_features.to(device), iq_labels.to(device)

    # sys.exit(0)
    output = classifier(iq_features)
    predicted_labels = torch.argmax(output, dim=1)
    
    print(f"Total correct: {torch.sum(iq_labels == predicted_labels)}")

    true = iq_labels.cpu()
    pred = predicted_labels.cpu()

    cm = confusion_matrix(true.numpy(), pred.numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=indices)
    disp.plot()
    plt.savefig('cm.png')