from sigmoid import sigmoid
import numpy as np

def sigmoidGradient(z):
#SIGMOIDGRADIENT returns the gradient of the sigmoid function
#evaluated at z
#   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
#   evaluated at z. This should work regardless if z is a matrix or a
#   vector. In particular, if z is a vector or matrix, you should return
#   the gradient for each element.

# The value g should be correctly computed by your code below.
    g = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of the sigmoid function evaluated at
#               each value of z (z can be a matrix, vector or scalar).
    g = 0
    g = sigmoid(z)
    g = g * (1-g)
    #g =  sigmoid(z) * (1- sigmoid(z))

# =============================================================
    
    return g