import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neural_network import oneHotEncodeOneCol, y_mse

class NnImg2Num(nn.Module):
    def __init__(self, batch_size=32, test_batch_size=100):
        super(NnImg2Num, self).__init__()  # initialize super class
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.train_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST('../data', train=True, download=True,
                              transform=tv.transforms.Compose([
                                  tv.transforms.ToTensor()
                              ])),
            batch_size=self.batch_size, shuffle=True)
        
        self.train_test_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST('../data', train=True, download=True,
                              transform=tv.transforms.Compose([
                                  tv.transforms.ToTensor()
                              ])),
            batch_size=self.test_batch_size, shuffle=True)

        # split train data in validation set too

        self.test_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST('../data', train=False, transform=tv.transforms.Compose([
                tv.transforms.ToTensor()
            ])),
            batch_size=self.test_batch_size, shuffle=False)

        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)

        layers = [
            nn.Linear(28*28, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 10),
        ]
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        
        return x

    def train(self, epochs=10):
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        it = 0
        i = 0
        train_mse = []
        train_accuracy = []
        test_mse = []
        test_accuracy = []
        r = []
        while it < epochs:
            for x, y in self.train_loader:
                N = x.shape[0]

                y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10).view(N, 1, -1)

                dim = 1
                for d in x.shape[1:]:
                    dim *= d

                x_transform = x.view(N, 1, dim)

                optimizer.zero_grad()
                y_hat = self.forward(x_transform)
                loss = criterion(y_hat, y_onehot)
                loss.backward()
                # put in closure as stopping point
                optimizer.step(closure=None)

                if (i % 1000 == 0):
                    print(i)
                    tm, ta = self.test_mse_accuracy()
                    # trm, tra = self.train_mse_accuracy()
                    trm = y_mse(y_onehot.view(N, -1), y_hat.view(N, -1))
                    tra = 1
                    test_mse.append(tm)
                    test_accuracy.append(ta)
                    train_mse.append(trm)
                    train_accuracy.append(tra)
                    r.append(i / self.train_size)

                i += 1
            it += 1

        if (i % 1000 == 0):
            print(i)
            tm, ta = self.test_mse_accuracy()
            trm, tra = self.train_mse_accuracy()
            # trm = y_mse(y_onehot.view(N, -1), y_hat.view(N, -1))
            # tra = 1
            test_mse.append(tm)
            test_accuracy.append(ta)
            train_mse.append(trm)
            train_accuracy.append(tra)
            r.append(i / self.train_size)

        return ((train_mse, train_accuracy), (test_mse, test_accuracy), r)

    def test_mse_accuracy(self):
        all_mse = []
        correct = 0
        total = 0
        for x, y in self.test_loader:
            y = y.view(-1, 1, 1)

            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(N, 1, dim)

            y_hat = self.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=2, keepdim=True)
            correct += torch.sum(y_hat_decoded == y).item()
            total += N

            y_hat = y_hat.view(N, -1)
            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            all_mse.append(y_mse(y_onehot, y_hat))
        
        mse = sum(all_mse) / len(all_mse)
        return (mse, correct / total)

    def test_accuracy(self):
        correct = 0
        total = 0
        for x, y in self.test_loader:
            y = y.view(-1, 1, 1)

            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(N, 1, dim)
            y_hat = self.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=2, keepdim=True)

            correct += torch.sum(y_hat_decoded == y).item()
            total += N

        return correct / total
    
    def test_mse(self):
        all_mse = []
        for x, y in self.test_loader:
            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(N, 1, dim)

            y_hat = self.forward(x_transform).view(N, -1)
            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            all_mse.append(y_mse(y_onehot, y_hat))
        
        mse = sum(all_mse) / len(all_mse)
        return mse

    def train_mse_accuracy(self):
        all_mse = []
        correct = 0
        total = 0
        for x, y in self.train_test_loader:
            y = y.view(-1, 1, 1)

            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(N, 1, dim)

            y_hat = self.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=2, keepdim=True)
            correct += torch.sum(y_hat_decoded == y).item()
            total += N

            y_hat = y_hat.view(N, -1)
            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            all_mse.append(y_mse(y_onehot, y_hat))
        
        mse = sum(all_mse) / len(all_mse)

        return (mse, correct / total)

    def train_accuracy(self):
        correct = 0
        total = 0
        for x, y in self.train_test_loader:
            y = y.view(-1, 1, 1)

            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(N, 1, dim)
            y_hat = self.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=2, keepdim=True)

            correct += torch.sum(y_hat_decoded == y).item()
            total += N

        return correct / total

    def train_mse(self):
        all_mse = []
        for x, y in self.train_test_loader:
            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(N, 1, dim)

            y_hat = self.forward(x_transform).view(N, -1)
            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            all_mse.append(y_mse(y_onehot, y_hat))
        
        mse = sum(all_mse) / len(all_mse)
        return mse