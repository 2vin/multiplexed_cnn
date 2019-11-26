# multiplexed_cnn
Training multiplexed convvolutional neural network on MNIST dataset

In order to train CNN on DIGITS images with Black background(Type-I data) and White background(Type-II data), we need to either train separate model for both the types, or combine the model for both the types into a single model. 
But what if we want to train a single model with two different branches for separate prediction on each type. We can do this by using a Multiplexing layer (https://github.com/danielegrattarola/keras-multiplexer) and a control signal.

Let's say-
Branch 1 - Images = B/W images | Control Signal = 0
Branch 2 - Images = W/B images | Control Signal = 1

This is how my Multiplexed CNN model on MNIST looks like - 
![model](https://github.com/2vin/multiplexed_cnn/blob/master/model.png)

We can use multiplexer to select final output of the model using the predefined control signal. This helps in training multiple branches of the network simultaneously by switching each branch using control signal during training time.

# Result
![result](https://user-images.githubusercontent.com/38634222/69677452-006f8180-10c9-11ea-820d-2b5344f81801.png)
