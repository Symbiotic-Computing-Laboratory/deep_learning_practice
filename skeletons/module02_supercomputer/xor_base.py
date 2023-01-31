'''
Deep Learning Demo: XOR

Command line version

Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import re

import argparse
import pickle

# Tensorflow 2.x way of doing things
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential

#################################################################
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################
def build_model(n_inputs, n_hidden, n_output, activation='elu', lrate=0.001):
    '''
    Construct a network with one hidden layer
    - Adam optimizer
    - MSE loss
    
    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of units in the hidden layer
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden and output units
    :param lrate: Learning rate for Adam Optimizer
    '''
    model = Sequential();
    model.add(InputLayer(input_shape=(n_inputs,)))
    model.add(Dense(n_hidden, use_bias=True, name="hidden", activation=activation))
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss='mse', optimizer=opt)
    
    # Generate an ASCII representation of the architecture
    print(model.summary())
    return model

def args2string(args):
    '''
    Translate the current set of arguments
    
    :param args: Command line arguments
    '''
    return "exp_%02d_hidden_%02d"%(args.exp, args.hidden)
    
    
########################################################
def execute_exp(args):
    '''
    Execute a single instance of an experiment.  The details are specified in the args object
    
    :param args: Command line arguments
    '''

    ##############################
    # Run the experiment
    # Create training set: XOR
    ins = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outs = np.array([[0], [1], [1], [0]])
    
    #Build Model
    model = build_model(ins.shape[1], args.hidden, outs.shape[1], activation='sigmoid')

    # Callbacks
    
    early_stopping_cb = # TODO

    # Describe arguments
    argstring = args2string(args)
    print("EXPERIMENT: %s"%argstring)
    
    # Only execute if we are 'going'
    if not args.nogo:
        # Training
        print("Training...")
        
        # Note: faking validation data set
        history = model.fit( #TODO
            )
        
        print("Done Training")
        
        # Save the training history
        fp = # TODO
        # TODO
        fp.close()

def display_learning_curve(fname):
    '''
    Display the learning curve that is stored in fname
    
    :param fname: Results file to load and dipslay
    
    '''
    
    # Load the history file and display it
    fp = #TODO
    # TODO
    fp.close()
    
    # Display
    plt.plot(history['loss'])
    plt.ylabel('MSE')
    plt.xlabel('epochs')

def display_learning_curve_set(dir, base):
    '''
    Plot the learning curves for a set of results
    
    :param base: Directory containing a set of results files
    '''
    # Find the list of files in the local directory that match base_[\d]+.pkl
    files = [f for f in os.listdir(dir) if re.match(r'%s.+.pkl'%(base), f)]
    files.sort()
    
    for f in files:
        with open("%s/%s"%(dir,f), "rb") as fp:
            history = pickle.load(fp)
            plt.plot(history['loss'])
    plt.ylabel('MSE')
    plt.xlabel('epochs')
    plt.legend(files)
    
def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='XOR Learner')
    
    parser.add_argument('--exp', type=int, default=0, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden', type=int, default=2, help='Number of Hidden Units')
    # BOOLEAN SWITCH
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    
    return parser

'''
This next bit of code is executed only if this python file itself is executed
(if it is imported into another file, then the code below is not executed)
'''
if __name__ == "__main__":
    # Parse the command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    if(n_physical_devices > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')

    # Do the work
    execute_exp(args)
