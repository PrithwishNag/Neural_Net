# Neural_Net
Simplistic and clean implementation of neural network, with back propagation (gradient descend), with scalable layers and easy to modify hyperparameters.

------------------------------------------------------------------------------------------------------------------------
Explain your implementation.
------------------------------------------------------------------------------------------------------------------------
Preprocessing:
All the functions for preprocessing the dataset are implemented in the 'Process' class. We have done the following
for preprocessing:
    1) Empty Fill - Filling empty values in the dataset with median values of that column. This function used for this is 'fill_median'.
    2) Removing outliers - all outliers of a particular column are removed which lie beyond the interquartile range of 0.25-0.75.
    3) MinMaxScaling - MinMax scaling is done using the inbuilt MinMaxScaling function of 'Preprocessing' library.

After cleaning, the data is split into training and testing data as per 80:20 ratio.
The neural network is then initialized, fit, predicted for the training and test set and evaluated.

The structure of the neural network is initialized in the 'NN' class which has the following functions : 
   
    1) __init__ : used for initializing the model, learning rate and number of epochs. The model is initialized
                  as 'None' in the beginning.
    2) fit : First, we initialize the input layer with the neurons. Then we assign the model of the neural network
             by initializing the 'Propogation' class. (what the propogation class does is explained later)
             Next, the hidden and output layers are added. We have built a generalized model and so any number of hidden
             and output layers can be added simply by calling the 'Layer' class with the desired number of neurons and
             activation function of your choice.
             Lastly, the model is updated and fit for each epoch.
    3) predict : This function feed forwards the X matrix to get desired binary output and saves that binary output.
    4) CM : this function is used for creating, calculating and displaying the confusion matrix and is used to print
            the evaluation metrics used : Accuracy, precision, recall and F1 score.

The 'Propogate' class is where the functionality of the neural network is implemented.
It consists of the following functions:
    1) delta : computing the delta vector
    2) update_weights : does the operation of computing gradient, updating weight and updating bias.
    3) feed_forward : mimics the logic for forward propogation i.e computing activation function for
                      each layer and passing it to subsequent layers
    4) back_propogate : mimics the logic for back propogation to reduce error
    5) trip : calls the front propogation, back propogation and update weights function respectively.

The 'Activation_fn' class defines the following two activation functions and is implemented accordingly.
     1) leaky_Relu
     2) sigmoid
     3) tanh

The 'Matrix_Operations' class defines all the matrix operations that are required.
     1) computing the hadamard product
     2) transpose the matrix

The 'Layer' class contains functions to initialize, add and store the activated sum for the neural network layers.
     1) add: adds the object of 'Layer' class to the list of layers and returns the current layer object.
     2) activated_sum: produces the activation(w*a - b), and updates the z of that layer(z = w*a-b)
We have used the 'Xaviers' initialization for weights.

-------------------------------------------------------------------------------------------------------------------------
What is the key feature of your design that makes it stand out?
-------------------------------------------------------------------------------------------------------------------------

The code is modularized in a way that makes it extremely readable as well as easy to modify.
One can modify the structure of the neural network according to his/her needs and quality of data.
The code is clean,flexible,readable and highly modularized.

We have implemented a generalized model where we have implemented the interface for adding, removing any number of
hidden layers by specifying the activation function and number of neurons for that layer.

We have also implemented an interface which gives user the option to have different activation functions. Currently, it
contains the implementation of the leaky_relu and sigmoid activation function.

Hyperparameter tuning is made easy as the hyperparameters need to be changed in only 1 place i.e the __init__ function
of the NN class.

------------------------------------------------------------------------------------------------------------------------
List of hyperparameters: 
------------------------------------------------------------------------------------------------------------------------
 - no. of epochs - 800
 - learning rate - 0.1
 -----------------------------------------------------------------------------------------------------------------------
Have you implemented something beyond the basics?
------------------------------------------------------------------------------------------------------------------------

- We have done self written implementations for effectively removing outliers and filling empty values of the dataset.
- We have created very easy to understand, modularized code. Anyone can add as many layers as needed, change hyperparameters without knowing the internal workings of the code.

-----------------------------------------------------------------------------------------------------------------------
Detailed steps to run your files.
-----------------------------------------------------------------------------------------------------------------------
Steps to execute the file : 
- Run the command python3 Neural_Net.py
