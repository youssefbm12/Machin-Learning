import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

# Useful values
    m = np.shape(X)[0]              #number of examples
    
# You need to return the following variables correctly 
    p = np.zeros(m);

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
    temp = np.ones((X.shape[0], 1)) # ADD BIAS TO X
    
    X = np.append(temp, X, axis=1)

    actions = sigmoid(np.dot(X, Theta1.T))
    
    temp = np.ones((actions.shape[0], 1)) # ADD BIAS TO actions
    
    actions = np.append(temp, actions, axis = 1)
    
    p = sigmoid(np.dot(actions, Theta2.T))

    p = np.argmax(p, axis = 1)



    return p+1

# =========================================================================
