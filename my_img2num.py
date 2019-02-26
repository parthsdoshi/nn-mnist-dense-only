import torch
import torchvision as tv

from neural_network import NeuralNetwork
from neural_network import oneHotEncodeOneCol, y_mse


class MyImg2Num:
    def __init__(self, batch_size=32, test_batch_size=100):
        self.nn = NeuralNetwork([28 * 28, 16, 16, 10], alpha=.1)
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

    def forward(self, x):
        self.nn.forward(x)

    def train(self, epochs=10):
        it = 0
        i = 0
        train_mse = []
        train_accuracy = []
        test_mse = []
        test_accuracy = []
        r = []
        while it < epochs:
            # print(f'epoch {it + 1}/{epochs}')
            for x, y in self.train_loader:
                # 10 for the number of digits in MNIST
                y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

                dim = 1
                for d in x.shape[1:]:
                    dim *= d

                x_transform = x.view(x.shape[0], dim)

                y_hat = self.nn.forward(x_transform)
                self.nn.backward(y_onehot)
                self.nn.updateParams()

                if (i % 1000 == 0):
                    print(i)
                    tm, ta = self.test_mse_accuracy()
                    # trm, tra = self.train_mse_accuracy()
                    trm = y_mse(y_onehot, y_hat)
                    tra = 1
                    test_mse.append(tm)
                    test_accuracy.append(ta)
                    train_mse.append(trm)
                    train_accuracy.append(tra)
                    r.append(i / self.train_size)

                i += 1
            it += 1

        if (i % 1000 == 0):
            tm, ta = self.test_mse_accuracy()
            trm, tra = self.train_mse_accuracy()
            # trm = y_mse(y_onehot, y_hat)
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
            y = y.view(-1, 1)
            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(x.shape[0], dim)

            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            y_hat = self.nn.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=1, keepdim=True)
            correct += torch.sum(y_hat_decoded == y).item()
            total += N

            all_mse.append(y_mse(y_onehot, y_hat))

        mse = sum(all_mse) / len(all_mse)
        return (mse, correct / total)

    def test_accuracy(self):
        correct = 0
        total = 0
        for x, y in self.test_loader:
            y = y.view(-1, 1)

            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(x.shape[0], dim)
            y_hat = self.nn.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=1, keepdim=True)

            correct += torch.sum(y_hat_decoded == y).item()
            total += N

        return correct / total

    def test_mse(self):
        all_mse = []
        for x, y in self.test_loader:
            y = y.view(-1, 1)
            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(x.shape[0], dim)

            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            all_mse.append(self.nn.mse(x_transform, y_onehot))

        mse = sum(all_mse) / len(all_mse)
        return mse

    def train_mse_accuracy(self):
        all_mse = []
        correct = 0
        total = 0
        for x, y in self.train_test_loader:
            y = y.view(-1, 1)
            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(x.shape[0], dim)

            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            y_hat = self.nn.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=1, keepdim=True)
            correct += torch.sum(y_hat_decoded == y).item()
            total += N

            all_mse.append(y_mse(y_onehot, y_hat))

        mse = sum(all_mse) / len(all_mse)
        return (mse, correct / total)

    def train_accuracy(self):
        correct = 0
        total = 0
        for x, y in self.train_test_loader:
            y = y.view(-1, 1)

            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(x.shape[0], dim)
            y_hat = self.nn.forward(x_transform)
            y_hat_decoded = torch.argmax(y_hat, dim=1, keepdim=True)

            correct += torch.sum(y_hat_decoded == y).item()
            total += N

        return correct / total

    def train_mse(self):
        all_mse = []
        for x, y in self.train_test_loader:
            y = y.view(-1, 1)
            N = x.shape[0]
            dim = 1
            for d in x.shape[1:]:
                dim *= d

            x_transform = x.view(x.shape[0], dim)

            y_onehot = oneHotEncodeOneCol(y.view(-1, 1), 10)

            all_mse.append(self.nn.mse(x_transform, y_onehot))

        mse = sum(all_mse) / len(all_mse)
        return mse