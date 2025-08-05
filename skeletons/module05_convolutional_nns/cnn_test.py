import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import argparse
import pickle
import keras

from keras import Sequential
from keras.layers import InputLayer
from keras.layers import Convolution2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, Dropout, SpatialDropout2D

import random
import re

import png

from core50 import *

import sklearn.metrics

##################
# Configure figure parameters

FONTSIZE = 18
FIGURE_SIZE = (10,4)
FIGURE_SIZE2 = (10,10)

plt.rcParams.update({'font.size': FONTSIZE})
plt.rcParams['figure.figsize'] = FIGURE_SIZE
# Default tick label size
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


#####


def load_data_sets(dataset_dir:str):
    ## File location
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

    ### Combine positives and negatives into a common data set
    outs_pos = np.append(np.ones((ins_pos.shape[0],1)), np.zeros((ins_pos.shape[0],1)), axis=1)
    outs_neg = np.append(np.zeros((ins_pos.shape[0],1)), np.ones((ins_pos.shape[0],1)), axis=1)

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

    ### Combine positives and negatives
    outs_pos_validation = np.append(np.ones((ins_pos_validation.shape[0], 1)), np.zeros((ins_pos_validation.shape[0], 1)), axis=1)
    outs_neg_validation = np.append(np.zeros((ins_pos_validation.shape[0], 1)), np.ones((ins_pos_validation.shape[0], 1)), axis=1)

    ins_validation = np.append(ins_pos_validation, ins_neg_validation, axis=0)
    outs_validation = np.append(outs_pos_validation, outs_neg_validation, axis=0)
    
    return ins, outs, ins_validation, outs_validation

def create_classifier_network(image_size, nchannels, n_classes, 
                              learning_rate=.0001, 
                              lambda_l2=None,                             # None or a float
                              p_dropout=None,
                              p_spatial_dropout=None,
                              n_filters=  [10],
                              kernel_size=[3],
                              pooling=[1],
                             n_hidden=[5]):
    
    if lambda_l2 is not None:
        # assume a float
        regularizer = tf.keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None
        
    
    model = Sequential()
    model.add(InputLayer(shape=(image_size[0], image_size[1], nchannels)))
   
    # convolutional layers
    for i, (n, s, p) in enumerate(zip(n_filters, kernel_size, pooling)):
        model.add(Convolution2D(filters=n,
                            kernel_size=s,
                            padding='same',
                            use_bias=True,
                            kernel_regularizer=regularizer,
                            name='C%d'%(i),
                            activation='elu'))
        
        if p_spatial_dropout is not None:
            model.add(SpatialDropout2D(p_spatial_dropout))
            
        if p > 1:
            model.add(MaxPooling2D(pool_size=p,
                           strides=p,
                           name='MP%d'%(i)))
        
    
    # Flatten
    model.add(GlobalMaxPooling2D())
    
    # Dense layers
    for i,n in enumerate(n_hidden):
        model.add(Dense(units=n,
                    activation='elu',
                    use_bias='True',
                    kernel_regularizer=regularizer,
                    name='D%d'%i))
        
        if p_dropout is not None:
            model.add(Dropout(p_dropout))
    
    # Output
    model.add(Dense(units=n_classes,
                    activation='softmax',
                    use_bias='True',
                    kernel_regularizer=regularizer,
                    name='output'))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                   amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['categorical_accuracy'])
    
    return model

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    
    # Training Parameters
    parser.add_argument('--lrate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    
    # Network Parameters
    parser.add_argument('--dropout', type=float, default=None, help="Dropout rate")
    parser.add_argument('--spatial_dropout', type=float, default=None, help="Spatial dropout rate")
    parser.add_argument('--l2', '--lambda_l2', type=float, default=None, help="L2 Regularization")
    parser.add_argument('--n_hidden', nargs='+', type=int, default=[5], help="Dense layer sizes")
    parser.add_argument('--n_filters', nargs='+', type=int, default=[10], help="Number of Conv filters")
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[3], help="Kernel sizes")
    parser.add_argument('--pooling', nargs='+', type=int, default=[0], help="Pooling sizes")

    # Dataset
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/core50', help="Learning rate")
    
    return parser


def generate_fname(args):
    strng = 'results'
    
    strng = strng + '_LR_%f'%(args.lrate)
    
    if args.dropout is not None:
        strng = strng + '_DR_%.1f'%(args.dropout)
        
    if args.spatial_dropout is not None:
        strng = strng + '_SDR_%.1f'%(args.spatial_dropout)
    
    if args.l2 is not None:
        strng = strng + '_L2_%f'%(args.l2)
        
    strng = strng + '_filters_' + '_'.join(str(n) for n in args.n_filters)
    
    strng = strng + '_kernels_' + '_'.join(str(n) for n in args.kernel_sizes)
    
    strng = strng + '_pooling_' + '_'.join(str(n) for n in args.pooling)
    
    strng = strng + '_hidden_' + '_'.join(str(n) for n in args.n_hidden)
    
    return strng
    
def execute_exp(args):
    ins, outs, ins_validation, outs_validation = load_data_sets(args.dataset)
    
    model = create_classifier_network((ins.shape[1], ins.shape[2]), 
                                      ins.shape[3], 2, 
                                  learning_rate=args.lrate,
                                  p_spatial_dropout=args.spatial_dropout,
                                  p_dropout=args.dropout,
                                  lambda_l2=args.l2,
                                  n_filters=args.n_filters,
                                  kernel_size=args.kernel_sizes,
                                  pooling=args.pooling,
                                  n_hidden=args.n_hidden)
    
    if args.verbose > 0:
        print(model.summary())
        
    if args.nogo:
        # Stop execution
        print("No execution")
        return
    
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True,
                                                      min_delta=0.01)

    # Training
    history = model.fit(x=ins, y=outs, epochs=args.epochs, 
                        verbose=(args.verbose > 1),
                        validation_data=(ins_validation, outs_validation), 
                        callbacks=[early_stopping_cb])
    
    # Generate results data
    results = {}
    results['args'] = args
    results['predict_validation'] = model.predict(ins_validation)
    #results['predict_validation_eval'] = model.evaluate(ins_validation)
        
    results['predict_training'] = model.predict(ins)
    #results['predict_training_eval'] = model.evaluate(ins)
    results['history'] = history.history

    # Save results
    fbase = generate_fname(args)
    
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    #if args.save_model:
    model.save("%s_model.keras"%(fbase))

    print(fbase)
    
if __name__ == "__main__":
    # Turn off GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Parse incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    # Do the work
    execute_exp(args)
