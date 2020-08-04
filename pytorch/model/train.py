import numpy as np
from sklearn import datasets
import torch
from model import IrisNet
from dataset import IrisDataset

def train(X, y, epochs=10, batch_size=16):
    dataset = IrisDataset(X, y)
    num_examples = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = IrisNet(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        num_correct = 0
        for i, (inputs, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            num_correct += (labels == outputs.argmax(1)).sum()
            loss.backward()
            optimizer.step()
        print(f"Finished: {epoch}, accuracy: {round(num_correct.float().numpy() / num_examples, 4) * 100}%")

    return model

def freeze(model, path='../frozen'):
    torch.save(model.state_dict(), f'{path}/model.pt')

if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    model = train(X, y, epochs=50)
    freeze(model)