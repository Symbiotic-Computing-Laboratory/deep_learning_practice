'''
Deep Learning Demo: XOR

Command line version

Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import sys
import argparse
import copy
import pickle
import random
import numpy as np
import os

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, InputLayer, Dense
from keras import Input, Model
from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib import colors

#################################################################
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################


def build_model(n_inputs:int, n_hidden:int, n_output:int, activation:str='elu', lrate:float=0.001)-> Sequential:
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
    model.add(#TODO)

    # Hidden layers
    model.add(Dense(#TODO))

    # Output layer
    model.add(Dense(#TODO))
    
    # Optimizer
    opt = keras.optimizers.Adam(#TODO, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(#TODO)
    
    # Generate an ASCII representation of the architecture
    if args.verbose >= 1:
        print(model.summary())

    return model


def args2string(args:argparse.ArgumentParser)->str:
    '''
    Translate the current set of arguments
    
    :param args: Command line arguments
    '''
    return # TODO
    
    
########################################################
def execute_exp(args:argparse.ArgumentParser):
    '''
    Execute a single instance of an experiment.  The details are specified in the args object
    
    :param args: Command line arguments
    '''

    ##############################
    # Run the experiment

    # Create training set: XOR
    ins = np.array()
    outs = np.array([)
    #####

    # Create the model
    model = build_model()

    # Callbacks

    # Stop training early if we stop making progress
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=100,
                                                      restore_best_weights=True,
                                                      min_delta=0.001,
                                                      monitor='loss')
    

    # Describe arguments
    argstring = args2string(args)
    print("EXPERIMENT: %s"%argstring)

    # Output pickle file
    fname_output = "results/xor_results_%s.pkl"%(argstring)

    # Does this file already exist?
    if os.path.exists(fname_output):
        print("File %s already exists."%fname_output)
        return
    
    # Only execute if we are 'going'
    if not args.nogo:
        # Training
        print("Training...")
        
        history = model.fit(# TODO
                            verbose=args.verbose>=2,
                            callbacks=[early_stopping_cb]
            )
        
        print("Done Training")
        
        # Save the training history
        with open(fname_output, "wb") as fp:
            pickle.dump(history.history, fp)
            pickle.dump(args, fp)

def display_learning_curve(fname:str):
    '''
    Display the learning curve that is stored in fname.
    As written, effectively assumes local execution
    (but could write the image out to a file)
    
    :param fname: Results file to load and dipslay
    
    '''
    
    # Load the history file and display it
    fp.close()
    
    # Display
    plt.plot(history['loss'])
    plt.ylabel('MSE')
    plt.xlabel('epochs')

def display_learning_curve_set(dir:str, base:str):
    '''
    Plot the learning curves for a set of results

    As written, effectively assumes local execution
    (but could write the image out to a file)
    
    :param base: Directory containing a set of results files
    '''
    # Find the list of files in the local directory that match base_[\d]+.pkl
    files = [f for f in os.listdir(dir) if re.match(r'%s.+.pkl'%(base), f)]
    files.sort()

    # Loop over the files
    for f in files:
        # Load the history data and plot it
        with open("%s/%s"%(dir,f), "rb") as fp:
            history = pickle.load(fp)
            plt.plot(history['loss'])

    # Finish figure
    plt.ylabel('MSE')
    plt.xlabel('epochs')
    plt.legend(files)
    
def create_parser()->argparse.ArgumentParser:
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='XOR Learner')

    parser.add_argument('--exp', type=int, default=0, help='Experiment number')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
                    
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    return parser

'''
This next bit of code is executed only if this python file itself is executed
(if it is imported into another file, then the code below is not executed)
'''
if __name__ == "__main__":
    # Parse the command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Do the work
    execute_exp(args)
