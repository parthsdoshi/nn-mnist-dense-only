import matplotlib.pyplot as plt
import numpy as np

from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num

train_batch = 1
test_batch = 10000
epochs = 1

my_nn = MyImg2Num(train_batch, test_batch)
train_error, test_error, epoch_scale = my_nn.train(epochs)
accuracy = my_nn.test_accuracy()
print(f'accuracy on test: {accuracy}')
mse = my_nn.test_mse()
print(f'mse on test: {mse}')

train_mse, train_accuracy = train_error
test_mse, test_accuracy = test_error

train_mse = np.array(train_mse)
train_accuracy = np.array(train_accuracy)

test_mse = np.array(test_mse)
test_accuracy = np.array(test_accuracy)

plt.plot(epoch_scale, train_mse, label='train mse')
plt.plot(epoch_scale, test_mse, label='test mse')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.title('MyNN MSE vs Epochs')
plt.legend()
plt.show()

py_nn = NnImg2Num(train_batch, test_batch)
train_error, test_error, epoch_scale = py_nn.train(epochs)
accuracy = py_nn.test_accuracy()
print(f'accuracy on test: {accuracy}')
mse = py_nn.test_mse()
print(f'mse on test: {mse}')

train_mse, train_accuracy = train_error
test_mse, test_accuracy = test_error

train_mse = np.array(train_mse)
train_accuracy = np.array(train_accuracy)

test_mse = np.array(test_mse)
test_accuracy = np.array(test_accuracy)

plt.plot(epoch_scale, train_mse, label='train mse')
plt.plot(epoch_scale, test_mse, label='test mse')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.title('PytorchNN MSE vs Epochs')
plt.legend()
plt.show()