'''
Deep Learning Practice: Convolutional Neural Network

Skeleton code: fill in the missing pieces

Andrew H. Fagg (andrewhfagg@gmail.com)

TODO:
- create_classifier_network()


'''
import sys
import argparse
import copy
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import time
import wandb
import socket
import matplotlib.pyplot as plt
from matplotlib import colors
import re
import png
import sklearn.metrics

# Data loader for the core 50 dataset
from core50 import *

# This is the keras 3 way of doing things
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Convolution2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, Dropout, SpatialDropout2D

#################################################################
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################

#####


def load_data_sets(dataset_dir:str)->(np.array, np.array, np.array, np.array, int):
    '''
    Load data sets for training/validation.

    :param dataset_dir: Directory in which the core50 data set is located
    :return: Tuple that contains the training ins/outs, validation ins/outs, and number of classes

    We have hard-coded a lot here just so we can bring in a small data set to play with.

    Data reference: https://vlomonaco.github.io/core50/

    Data and Directory Organization:
    - 10 object classes
    - 50 different object instances o01 ... o50 (5 for each class).  o01 ... o05 are class 1, etc.
    - For each object instance, there are 11 different background conditions
    - For each background condition, there are 300 frames from a movie as that object is being manipulated

    For this implementation, we are loading one object (o21; a can) to serve as the positive examples and
    another object (o41; a cup) as the negative examples.  For our training set, we will take only the
    image indices that end in 0 (so 300/10=30 images).

    For validation data, we will take images ending in '5' for objects o22 (can, positive) and
    o42 (cup, negative).

    Note that this is a really "dumbed-down" version of an image recognition problem: we are not training a model
    that can distinguish between all cans and cups.

    One can edit this function to work with a much larger set of objects and conditions
    (not ideal for general implementations, but will serve our purposes).
    '''

    ## Dataset location (we are using the 128 x 128 images)
    directory_base = '%s/core50_128x128'%dataset_dir

    # Training set: define which files to load for each object
    #test_files = '.*[05].png'
    test_files = '.*[0].png'

    ### Positive cases
    # Define which objects to load
    #object_list = ['o25', 'o22', 'o23', 'o24']
    object_list = ['o21']

    # Define which conditions to load
    #condition_list = ['s1', 's2', 's3', 's4', 's5', 's7', 's8', 's9', 's10', 's11']
    #condition_list = ['s1', 's2', 's3', 's4']
    condition_list = ['s1']

    # Load all of the objects/condition
    ins_pos = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list, test_files)

    ### Negative cases
    # Define which objects to load
    #object_list2 = ['o45', 'o42', 'o43', 'o44']
    object_list2 = ['o41']
    ins_neg = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list2, test_files)

    ### Create the labels: first set of images are 'class 1'; second set is 'class 0'
    outs_pos = np.ones((ins_pos.shape[0],))
    outs_neg = np.zeros((ins_neg.shape[0],))

    # Combine the positive and negative examples
    ins = np.append(ins_pos, ins_neg, axis=0)
    outs = np.append(outs_pos, outs_neg, axis=0)

    ########################################################################
    # Validation set
    # Define which files to load for each object
    test_files = '.*[5].png'

    ### Positives
    # Define which objects to load
    object_list = ['o22']
    #object_list = ['o21']
    condition_list = ['s2']

    # Load the positives
    ins_pos_validation = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list, test_files)

    ### Negatives
    # Define objects
    object_list2 = ['o42']
    #object_list2 = ['o41']

    # Load the negative cases
    ins_neg_validation = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list2, test_files)

    ### Create the labels
    outs_pos_validation = np.ones((ins_pos_validation.shape[0],))
    outs_neg_validation = np.zeros((ins_pos_validation.shape[0],))

    # Combine the positives from the negatives
    ins_validation = np.append(ins_pos_validation, ins_neg_validation, axis=0)
    outs_validation = np.append(outs_pos_validation, outs_neg_validation, axis=0)

    # Return the training and validation images and their labels
    return ins, outs, ins_validation, outs_validation, 2

def create_classifier_network(image_size:(int,int),
                              nchannels:int=3,
                              n_classes:int=2, 
                              learning_rate:float=.0001, 
                              lambda_l2:float=None,
                              p_dropout:float=None,
                              p_spatial_dropout:float=None,
                              n_filters:[int]=  [10],
                              kernel_size:[int]=[3],
                              pooling:[int]=[1],
                              n_hidden:[int]=[5],
                              activation_internal:str='elu',
                              activation_output:str='softmax',
                              loss:str='sparse_categorical_crossentropy',
                              metrics:[str]=['sparse_categorical_accuracy']):
    

    '''
    Create the classifier CNN network.

    :param image_size: 2-tuple (rows, col)
    :param nchannels: number of image channels (typeically 3: red, green, blue)
    :param n_classes: number of classes
    :param learning_rate: float
    :param lambda_l2: L2 regularization parameter
    :param p_dropout: Dropout parameter for fully connected layers
    :param p_spatial_dropout: Spatial dropout parameter for CNN layers
    :param n_filters: Number of filters for each convolutional module
    :param kernel_size: Kernel size for each convolutional module
    :param pooling: Pooling size for each convolutional module
    :param n_hidden: Number of hidden units in each fully connected layer
    :activation_internal: Activation function used for all internal layers
    :activation_output: Activation function used for output layer
    :loss: Loss function
    :metrics: List of metrics to measure performance against
    
    '''
    
    # L2 Regularization support
    if lambda_l2 is not None:
        # assume a float
        regularizer = keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None
        
    # Construct the model
    model = Sequential()

    # TODO: implement the CNN

    # Output
    model.add(Dense(units=n_classes,
                    activation=activation_output,
                    use_bias='True',
                    name='output'))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                   amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)
    
    return model


def create_parser()->argparse.ArgumentParser:
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--force', action='store_true', help='Execute experiment even if there is already a results file')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    
    # Training Parameters
    parser.add_argument('--lrate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs to wait before Early Stopping')
    parser.add_argument('--monitor', type=str, default='loss', help="Monitor variable for Early Stopping")
    parser.add_argument('--min_delta', type=float, default=0.01, help="Minimum delta for Early Stopping")
    
    # Network Parameters
    parser.add_argument('--dropout', type=float, default=None, help="Dropout rate")
    parser.add_argument('--spatial_dropout', type=float, default=None, help="Spatial dropout rate")
    parser.add_argument('--lambda_l2', '--l2', type=float, default=None, help="L2 Regularization")
    parser.add_argument('--n_hidden', nargs='+', type=int, default=[5], help="Dense layer sizes")
    parser.add_argument('--n_filters', nargs='+', type=int, default=[10], help="Number of Conv filters")
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[3], help="Kernel sizes")
    parser.add_argument('--pooling', nargs='+', type=int, default=[0], help="Pooling sizes")

    # Dataset
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/core50', help="Learning rate")

    # Misc
    parser.add_argument('--render', action='store_true', help='Generate a PNG of the network')

    # WandB support                                                                                                                   
    parser.add_argument('--wandb', action='store_true', help='Turn on reporting to WandB')
    parser.add_argument('--wandb_project', type=str, default='XOR', help='WandB project name')
    parser.add_argument('--wandb_run_label', type=str, default='version0', help='WandB project name')

    return parser


def generate_fname(args:argparse.ArgumentParser)->str:
    '''
    Translate the parsed args into a file name that describes the specific experiment

    :param args: Argument object from the parser
    :return: String that describes this experiment
    '''
    strng = 'cnn'

    if args.dropout is not None:
        strng = strng + '_DR_%.2f'%(args.dropout)
        
    if args.spatial_dropout is not None:
        strng = strng + '_SDR_%.2f'%(args.spatial_dropout)
    
    if args.lambda_l2 is not None:
        strng = strng + '_L2_%f'%(args.lambda_l2)
        
    strng = strng + '_filters_' + '_'.join(str(n) for n in args.n_filters)
    
    strng = strng + '_kernels_' + '_'.join(str(n) for n in args.kernel_sizes)
    
    strng = strng + '_pooling_' + '_'.join(str(n) for n in args.pooling)
    
    strng = strng + '_hidden_' + '_'.join(str(n) for n in args.n_hidden)
    
    return strng

    
def execute_exp(args:argparse.ArgumentParser):
    '''
    Do all of the work

    :param args: Argument data structure
    '''

    # Load the data
    ins, outs, ins_validation, outs_validation, n_classes = load_data_sets(args.dataset)

    # Create the model
    model = create_classifier_network((ins.shape[1], ins.shape[2]), 
                                      ins.shape[3],
                                      n_classes, 
                                      learning_rate=args.lrate,
                                      p_spatial_dropout=args.spatial_dropout,
                                      p_dropout=args.dropout,
                                      lambda_l2=args.lambda_l2,
                                      n_filters=args.n_filters,
                                      kernel_size=args.kernel_sizes,
                                      pooling=args.pooling,
                                      n_hidden=args.n_hidden)

    # Optionally print the model summary
    if args.verbose > 0:
        print(model.summary())

    # Results pickle file name
    fbase = generate_fname(args)
    results_fname = 'results/%s_results.pkl'%(fbase)

    # Does this file already exist?
    if not args.force and os.path.exists(results_fname):
        print("File %s already exists."%results_fname)
        return

    if args.nogo:
        # Stop execution
        print("No execution")
        return
        
    #############################################
    # Plot the model to a file
    if args.render:
        plot_model(model, to_file='results/%s_model_plot.png'%fbase, show_shapes=True, show_layer_names=True)
    
    # Callbacks
    cbs = []
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta,
                                                      monitor=args.monitor)
    cbs.append(early_stopping_cb)

    #############################################
    # WandB
    
    if args.wandb:
        # Connect to WandB
        wandb.init(project=args.wandb_project,
                   name=f'{args.wandb_run_label}',
                   notes=fbase,
                   config=vars(args))

        if args.render:
            # Log the architecture diagram to WandB
            wandb.log({'architecture': wandb.Image('results/%s_model_plot.png'%fbase)})

        # Log hostname
        wandb.log({'hostname': socket.gethostname()})

        # Create WandB callback
        wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
        cbs.append(wandb_metrics_cb)

    #################################
    # Training
    history = model.fit(x=ins, y=outs, epochs=args.epochs, 
                        verbose=(args.verbose >= 3),
                        validation_data=(ins_validation, outs_validation), 
                        callbacks=cbs)
    
    #################################
    # Generate results data
    results = {}
    wandb_results = {}

    results['args'] = args
    results['predict_validation'] = model.predict(ins_validation)
    results['predict_validation_eval'] = model.evaluate(ins_validation, outs_validation)
    
    wandb_results['final_val_loss'] = results['predict_validation_eval'][0]
    wandb_results['final_val_accuracy'] = results['predict_validation_eval'][1]

    if args.verbose >= 2:
        # Report the prediction for the validation set

        # Probabilities
        probs = results['predict_validation']
        # Predicted class is the class with the highest probability
        preds = np.argmax(probs, axis=-1)
        # Loop over the examples: report true, prediction, and probabilities
        print("Ground Truth\tPrediction\tProbabilities")
        for i in range(outs_validation.shape[0]):
            print(f'{int(outs_validation[i]):d}\t\t{int(preds[i]):d}\t\t{probs[i,0]:.3f}\t{probs[i,1]:.3f}')
            
            
        
    results['predict_training'] = model.predict(ins)
    results['predict_training_eval'] = model.evaluate(ins, outs)
    wandb_results['final_training_loss'] = results['predict_training_eval'][0]
    wandb_results['final_training_accuracy'] = results['predict_training_eval'][1]

    
    results['history'] = history.history

    #############################################
    if args.wandb:
        # Write final performance measures 
        wandb.log(wandb_results)

        # Close connection to WandB
        wandb.finish()

    ############################################
    # Save results
    results['fname_base'] = fbase
    with open(results_fname, "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        model.save("results/%s_model.keras"%(fbase))

if __name__ == "__main__":
    # Turn off GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Parse incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    # Do the work
    execute_exp(args)
