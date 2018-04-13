import time
import random
import numpy as np
from utils import *
from transfer_functions import * 


class NeuralNetwork(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, transfer_f=sigmoid, transfer_df=dsigmoid):
        """
        input_layer_size: number of input neurons
        hidden_layer_size: number of hidden neurons
        output_layer_size: number of output neurons
        iterations: number of iterations
        learning_rate: initial learning rate
        """

        # initialize transfer functions
        self.transfer_f = transfer_f
        self.transfer_df = transfer_df

        # initialize layer sizes
        self.input_layer_size = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden_layer_size = hidden_layer_size+1 # +1 for the bias node in the hidden layer 
        self.output_layer_size = output_layer_size

        # initialize arrays for activations
        self.u_hidden = np.zeros((1, self.hidden_layer_size-1))
        self.u_output = np.zeros((1, self.output_layer_size))

        # initialize arrays for outputs
        self.o_input = np.ones((1, self.input_layer_size))
        self.o_hidden = np.ones((1, self.hidden_layer_size))
        self.o_output = np.ones((1, self.output_layer_size))

        # initialize arrays for partial derivatives according to activations
        self.dE_du_hidden = np.zeros((1, self.hidden_layer_size-1))
        self.dE_du_output = np.zeros((1, self.output_layer_size))

        # create randomized weights Yann LeCun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input_layer_size ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input_layer_size, self.hidden_layer_size-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden_layer_size, self.output_layer_size)) / np.sqrt(self.hidden_layer_size)

    def weights_init(self,wi=None,wo=None):
        input_range = 1.0 / self.input_layer_size ** (1/2)
        if wi is not None:
            self.W_input_to_hidden = wi # weights between input and hidden layers
        else:
            self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input_layer_size, self.hidden_layer_size-1))
        if wo is not None:
            self.W_hidden_to_output = wo # weights between hidden and output layers
        else:
            self.W_hidden_to_output = np.random.uniform(size = (self.hidden_layer_size, self.output_layer_size)) / np.sqrt(self.hidden_layer_size)

    def train(self, data, validation_data, iterations=50, learning_rate=5.0, verbose=False):
        start_time = time.time()
        training_accuracies = []
        validation_accuracies = []
        MSE = []
        inputs  = data[0]
        targets = data[1]
        best_val_acc = 100*self.predict(validation_data)/len(validation_data[0])
        best_i2h_W = self.W_input_to_hidden
        best_h2o_W = self.W_hidden_to_output
        for it in range(iterations):
            self.feedforward(inputs)
            self.backpropagate(targets, learning_rate=learning_rate)
            error = targets - self.o_output
            error *= error
            MSE.append(np.sum(error)/len(data[0]))
            training_accuracies.append(100*self.predict(data)/len(data[0]))
            validation_accuracies.append(100*self.predict(validation_data)/len(validation_data[0]))
            if validation_accuracies[-1] > best_val_acc:
                best_i2h_W = self.W_input_to_hidden
                best_h2o_W = self.W_hidden_to_output
            if verbose:
                print("[Iteration %2d/%2d]  -Training_Accuracy:  %2.2f %%  -Validation_Accuracy: %2.2f %%  -time: %2.2f " %(it+1, iterations,
                                                            training_accuracies[-1], validation_accuracies[-1], time.time() - start_time))
                print("    - MSE:", np.sum(error)/len(targets))
        print("Training time:", time.time()-start_time)
        plot_train_val(range(1, iterations+1), training_accuracies, validation_accuracies, "Accuracy", MSE)

    def train_on_each_image(self, data, validation_data, iterations=50, learning_rate=5.0, verbose=False):
        start_time = time.time()
        training_accuracies = []
        validation_accuracies = []
        
        
        MSE = []
        MSE_validation = []
        
        best_val_acc = 100*self.predict(validation_data)/len(validation_data[0])
        best_i2h_W = self.W_input_to_hidden
        best_h2o_W = self.W_hidden_to_output
        
        for it in range(iterations):
            # Training with data
            errors = []
            for image in range(len(data[0])):
                targets = data[1][image]
                #print(len(targets))
                self.feedforward(inputs = [data[0][image]])
                self.backpropagate(targets, learning_rate=learning_rate)
                # Computation of the training errors
                error = [targets] - self.o_output
                error *= error
                errors.append(error[0])
            training_accuracies.append(100*self.predict_on_each_image([[data[0][image]],[data[1][image]]])/len(data[0][image]))
            MSE.append(np.sum(errors)/len(data[0]))
            
            errors_validation = []
            for image in range(len(validation_data[0])):
                targets_validation = validation_data[1][image]
                error_validation = [targets_validation] - self.o_output
                error_validation *= error_validation
                errors_validation.append(error_validation[0])
            MSE_validation.append(np.sum(errors_validation)/len(validation_data[0]))
            validation_accuracies.append(100*self.predict_on_each_image([[validation_data[0][image]],[validation_data[1][image]]])/len(validation_data[0][image]))
            
            if validation_accuracies[-1] > best_val_acc:
                best_i2h_W = self.W_input_to_hidden
                best_h2o_W = self.W_hidden_to_output
            if verbose:
                print("[Iteration %2d/%2d]  -Training_Accuracy:  %2.2f %%  -Validation_Accuracy: %2.2f %%  -time: %2.2f " %(it+1, iterations,
                                                            training_accuracies[-1], validation_accuracies[-1], time.time() - start_time))
                print("    - MSE:", np.sum(error)/len(targets))
        print("Training time:", time.time()-start_time)
        plot_train_val(range(1, iterations+1), training_accuracies, validation_accuracies, "Accuracy", MSE, MSE_validation)
        
        
    def train_xe(self, data, validation_data, iterations=50, learning_rate=5.0, verbose=False):
        start_time = time.time()
        training_accuracies = []
        validation_accuracies = []
        errors = []
        xes = []
        MSE = []
        X_entropy = []
        inputs  = data[0]
        targets = data[1]
        best_val_acc = 100*self.predict_xe(validation_data)/len(validation_data[0])
        best_i2h_W = self.W_input_to_hidden
        best_h2o_W = self.W_hidden_to_output
        for it in range(iterations):
            self.feedforward_xe(inputs)
            self.backpropagate_xe(targets, learning_rate=learning_rate)
            xe = targets*np.log(self.o_output)*(-1)
            error = targets - self.o_output
            error *= error
            MSE.append(np.sum(xe)/len(data[0]))
            X_entropy.append(np.sum(error)/len(data[0]))
            training_accuracies.append(100*self.predict_xe(data)/len(data[0]))
            validation_accuracies.append(100*self.predict_xe(validation_data)/len(validation_data[0]))
            if validation_accuracies[-1] > best_val_acc:
                best_i2h_W = self.W_input_to_hidden
                best_h2o_W = self.W_hidden_to_output
            if verbose:
                print("[Iteration %2d/%2d]  -Training_Accuracy:  %2.2f %%  -Validation_Accuracy: %2.2f %%  -time: %2.2f " %(it+1, iterations,
                                                            training_accuracies[-1], validation_accuracies[-1], time.time() - start_time))
                print("    - MSE:", np.sum(error)/len(targets))
                print("    - X-Entropy:", np.sum(xe)/len(targets))
        print("Training time:", time.time()-start_time)
        self.W_input_to_hidden = best_i2h_W
        self.W_hidden_to_output = best_h2o_W
        plot_train_val(range(1, iterations+1), training_accuracies, validation_accuracies, "Accuracy", MSE, X_entropy = X_entropy)
     

    def predict_on_each_image(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        self.feedforward(test_data[0])
        answer = np.argmax(test_data[1], axis=1)
        prediction = np.argmax(self.o_output, axis=1)
        count = len(test_data[0][0]) - np.count_nonzero(answer - prediction)
        return count 
    
    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        self.feedforward(test_data[0])
        answer = np.argmax(test_data[1], axis=1)
        prediction = np.argmax(self.o_output, axis=1)
        count = len(test_data[0]) - np.count_nonzero(answer - prediction)
        return count
    
    def predict_xe(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        self.feedforward_xe(test_data[0])
        answer = np.argmax(test_data[1], axis=1)
        prediction = np.argmax(self.o_output, axis=1)
        count = len(test_data[0]) - np.count_nonzero(answer - prediction)
        return count

