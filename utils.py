import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt


def load_data():
    np.random.seed(1990)
    print("Loading MNIST data .....")

    # Load the MNIST dataset
    with gzip.open('Data/mnist.pkl.gz', 'r') as f:
        # u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        # train_set, valid_set, test_set = u.load()
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        train_set = [train_set[0].tolist(), [[1 if j == train_set[1][i] else 0 for j in range(10)] for i in np.arange(len(train_set[0]))]]
        valid_set = [valid_set[0].tolist(), [[1 if j == valid_set[1][i] else 0 for j in range(10)] for i in np.arange(len(valid_set[0]))]]
        test_set = [test_set[0].tolist(), [[1 if j == test_set[1][i] else 0 for j in range(10)] for i in np.arange(len(test_set[0]))]]
    print("Done.")
    return train_set, valid_set, test_set

   
def plot_curve(t,s,metric):
    plt.plot(t, s)
    plt.ylabel(metric) # or ERROR
    plt.xlabel('Epoch')
    plt.title('Learning Curve_'+str(metric))
    #curve_name=str(metric)+"LC.png"
    #plt.savefig(Figures/curve_name)
    plt.show()
    
def plot_train_val(t, st, sv, metric, MSE_st, MSE_sv=None, X_entropy = None):
    if X_entropy == None :
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 8))
    ax1.plot(t, st, label='Accuracy on training set')
    ax1.plot(t, sv, label='Accuracy on validation set')
    ax1.legend(['Accuracy on training set', 'Accuracy on validation set'], bbox_to_anchor=(0.5, 0., 0., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    ax1.set_ylabel(metric) # or ERROR
    ax1.set_xlabel('Epoch')
    ax1.set_title('Learning Curve: '+str(metric))
    #curve_name=str(metric)+"LC.png"
    #plt.savefig(Figures/curve_name)
    ax2.plot(t, MSE_st, label='MSE')
    if MSE_sv:
        ax2.plot(t, MSE_sv, label='MSE on validation set')
    ax2.legend(['MSE on training set', 'MSE on validation set'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax2.set_ylabel('MSE') # or ERROR
    ax2.set_xlabel('Epoch')
    ax2.set_title('Learning Curve: MSE')
    if X_entropy:
        ax2.legend(['MSE on training set'], loc='upper right')
        ax3.plot(t, X_entropy, label='X_entropy on training set')
        ax3.legend(['X_entropy'], loc='upper right')
        ax3.set_ylabel('X_entropy') # or ERROR
        ax3.set_xlabel('Epoch')
        ax3.set_title('Learning Curve: X_entropy')
    plt.show()

