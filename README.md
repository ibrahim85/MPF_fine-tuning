# MPF_fine-tuning

This folder is for MPF of fully-visible graph network. 
The algorithm first train a stacked auto encoder with a softmat classifier for MNIST first, 
and then conduct finetuning with MPF. 

The objective function of MPF here is to minimize the exponentioal energy difference among the data samples and non data samples which 
are 1-bit away for the data-samples. $$min sum_{i,j} exp(E(i) - E(j))$$

For running the code, please:

(1): Install Theano and respective packages used in the code, i.e., numpy, scipy, sklearn, IMAGE, etc. 

(2): Clone or download the code, also the dataset mnist.pkl.gz, and then run $ python3 MPF_Mnist.py. 

For running the code on GPU machine, please:
Specify floatX = float32, device = gpu in the .theanorc configuration file. 

