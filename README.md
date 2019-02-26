## Code explanation
First, I created the Neural Network that I thought would be able to fit the MNist Dataset. I settled on `[28*28, 16, 16, 10]`. The 28*28 signifies the input layer which is that large because that's how much information is held inside each 28x28 image of MNist. I chose two 16 sized hidden layers as a guess that turned out to work pretty well in my case. Lastly, there are 10 digits that the MNist dataset classifies and thus we need 10 output neurons.

In order to work with the labels that the dataset provides, we need to one-hot encode it so that the number 6 would turn into `[0,0,0,0,0,0,1,0,0,0,0]`. This is because that is what we want our neural network to output in its output layer. I then added functions to test for accuracy and MSE against both the train and test datasets. I then ran both my implementation of this neural net and pytorch's and had them both return arrays of MSE for train and test. I then plotted these in my parth_test.py file.

Also, I'm using 0.1 as my learning rate and a batch size of 1. I landed on these hyperparameters after a lot of testing. That learning rate seemed to perform the best jumps over about 10 tests that I did. I also found that using a batch size of 1 and training for just 1 epoch was good enough for my neural net to converge.

Overall, I'm seeing about 0.2 MSE and about 90% accuracy on the test dataset for both my nn implementation and pytorch's implementation.
