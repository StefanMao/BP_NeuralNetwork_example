import numpy as np
import math
import sys

class Neural_Network():


    def __init__(self,x):

        self.learning_rate = 0.03
        self.epoch = 20000

        Input_layer_neurons= x
        print Input_layer_neurons
        Hidden_layer_neurons= 10
        Output_neurons = 1


        #weight random

        self.Weight_Hidden=np.random.uniform(size=(Input_layer_neurons, Hidden_layer_neurons))
        self.Bias_Hidden=np.random.uniform(size=(1, Hidden_layer_neurons))
        self.Weight_output=np.random.uniform(size=(Hidden_layer_neurons, Output_neurons))
        self.Bias_output=np.random.uniform(size=(1, Output_neurons))




    def sigmoid(self,x):
        return np.tanh(x)

    def sigmoid_derivative(self,x):
        return 1.0-x**2


    def train(self,training_set_input,training_set_output):


        for i in range(self.epoch):

            Hidden_layer_input1 = np.dot(training_set_input, self.Weight_Hidden)
            Hidden_layer_input = Hidden_layer_input1 + self.Bias_Hidden
            Hidden_layer_activations = self.sigmoid(Hidden_layer_input)

            Output_layer_input1 = np.dot(Hidden_layer_activations, self.Weight_output)
            Output_layer_input = Output_layer_input1 + self.Bias_output
            self.output = self.sigmoid(Output_layer_input)
            


            # BackPropagation
            Error = training_set_output - self.output
            slope_output_layer = self.sigmoid_derivative(self.output)
            slope_hidden_layer = self.sigmoid_derivative(Hidden_layer_activations)

            delta_of_output = Error * slope_output_layer
            Error_at_Hidden_layer= delta_of_output.dot(self.Weight_output.T)
            delta_of_hidden=Error_at_Hidden_layer*slope_hidden_layer

            self.Weight_output+=Hidden_layer_activations.T.dot(delta_of_output)*self.learning_rate
            self.Bias_output+=np.sum(delta_of_output,axis=0,keepdims=True)*self.learning_rate
            self.Weight_Hidden+=training_set_input.T.dot(delta_of_hidden)*self.learning_rate
            self.Bias_Hidden+=np.sum(delta_of_hidden,axis=0,keepdims=True)*self.learning_rate


    def think(self,predict_test_input):

        Hidden_layer_input1 = np.dot(predict_test_input, self.Weight_Hidden)
        Hidden_layer_input = Hidden_layer_input1 + self.Bias_Hidden
        Hidden_layer_activations = self.sigmoid(Hidden_layer_input)

        Output_layer_input1 = np.dot(Hidden_layer_activations, self.Weight_output)
        Output_layer_input = Output_layer_input1 + self.Bias_output
        self.output = self.sigmoid(Output_layer_input)
        return self.output

    def train_sample(self,x):

       return np.exp(-x)*math.cos((-3)*x)










if __name__=='__main__':

    training_set_input = np.array(
        [[0], [0.2], [0.4], [0.6], [0.8], [1], [1.2], [1.4], [1.6], [1.8], [2], [2.2], [2.4], [2.6], [2.8],
        [3], [3.2], [3.4], [3.6], [3.8], [4]])

    training_set_output = np.array(
       [[1],[0.675727649537],[0.242895666662],[-0.124691153313], [-0.331332354351],[-0.364197886413],[-0.270098444483],
        [-0.120896830082],[0.0176657400845],[0.104914026755],[0.129944917699],[0.105288772368],[0.0551883861353],
         [0.00400746214925],[-0.0315779755774],[-0.0453625046405],[-0.0401380472329],[-0.0238373804285],
         [-0.00530981642719],[0.00880269439855],[0.015455724383]])

    #training_set_input = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0],[0, 1, 1],[0, 0, 0],[1,1,1]])
    #training_set_output = np.array([[1], [1], [0],[0],[0],[1]])

    Neural_Network=Neural_Network(training_set_input.shape[1])

    Neural_Network.train(training_set_input,training_set_output)
    print("output of Forward Propogation:\n{}".format(Neural_Network.output))
    print("wout,bout of Backpropagation:\n{},\n{}".format(Neural_Network.Weight_output,Neural_Network.Bias_output))

    print "test"
    x=0
    for i in range(401):
        step= Neural_Network.think(np.array([x]))
        print '\n'.join(''.join(str(cell) for cell in row )for row in step)
        x+=0.01









