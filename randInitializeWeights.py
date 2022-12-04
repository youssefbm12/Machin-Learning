import numpy as np

def randInitializeWeights(L_in, L_out):
#RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
#incoming connections and L_out outgoing connections
#   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
#   of a layer with L_in incoming connections and L_out outgoing 
#   connections. 
#
#   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
#   the first row of W handles the "bias" terms
#

# You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

# ====================== YOUR CODE HERE ======================
# Instructions: Initialize W randomly so that we break the symmetry while
#               training the neural network.
#
# Note: The first row of W corresponds to the parameters for the bias units
#

   
    temp = np.ones((W.shape[0], 1)) # ADD BIAS TO X
    
    W = np.append(temp, W, axis=1)
    epsilon = 0.12 
    W = 2 * np.random.rand(L_out , L_in) * epsilon - epsilon
# =========================================================================

    return W
