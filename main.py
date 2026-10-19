import numpy as np

def sg(x):
    return (1/(1+np.exp(-x)))

class neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self,x):
        out = np.dot(self.weights,x) + self.bias
        return sg(out)



class newnet:
    def __init__(self):
        w = np.array([2,3])
        b =-1
        self.h1 = neuron(w,b)
        self.h2 = neuron(w,b)
        self.o1 = neuron(w,b)

    def feedforward(self, x):
        _h1 = self.h1.feedforward(x)
        _h2 = self.h1.feedforward(x)
        _o1 = self.o1.feedforward(np.array([_h1,_h2]))
        return _o1

n = newnet()
x = np.array([2,5])
print(n.feedforward(x))