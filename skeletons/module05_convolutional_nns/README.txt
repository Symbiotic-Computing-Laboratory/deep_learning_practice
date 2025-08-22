Image recognition example with the core50 data set.  For this example,
we are only loading in a small subset of the available data.

Basic execution on schooner:

1. Get a bash shell on a compute node

srun --partition normal --pty bash

2. Activate a keras3/tensorflow environment

. /home/fagg/tf_setup.sh
conda activate dnn

3. Execute.  I recommend implementing your own cnn_classifier from the
skeleton, but this should work out of the box:

python cnn_classifier_solution.py @network.txt @experiment.txt @wandb.txt
