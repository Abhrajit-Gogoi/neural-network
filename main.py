import numpy as np

def sg(x,d=False):
    if d == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

## initialising the weights (randomly and nor using a random module)
w1 = np.array(([0.3,0.2],[-0.2,0.022]))   # these are wights for connection between input and first hidden layer

w2 = np.array([-0.03,0.2])  # connections between first layer and the output neuron


# defining loss function
def mse(y_t, y_p):
    return (y_t-y_p)**2



# calculating the partial derivatives
def pardev(mse,h,o,i):
    # first layer
    d1 = mse*w2[0]*sg(o,True)*i[0]*sg(sg(h[0]),True)
    d2 = mse*w2[0]*sg(o,True)*i[1]*sg(sg(h[0]),True)
    d3 = mse*w2[1]*sg(o,True)*i[0]*sg(sg(h[1]),True)
    d4 = mse*w2[1]*sg(o,True)*i[1]*sg(sg(h[1]),True)

    # second layer
    d5 = mse*h[0]*sg(o,True)
    d6 = mse*h[1]*sg(o,True)
    out = np.array([d1,d2,d3,d4,d5,d6])
    
    return out

# defining the class and training bg
class net:
    def __init__(self,w1,w2):
        self.w1 = w1
        self.w2 = w2
    def train(self,x, y_t,epoch):
        n=0.1
        for j in range(epoch):
            for i in range(len(x)):
                h = np.dot(self.w1,x[i])
                o = np.dot(self.w2,sg(h))
                ms = -2*(y_t[i]-sg(o))
                lr = n*pardev(ms,sg(h),sg(o),x[i])
                self.w1 = self.w1 - np.array(([lr[0],lr[1]],[lr[2],lr[3]]))
                self.w2 -= np.array([lr[4],lr[5]])
            if j % 100 == 0:
                print(f"epoch: {j}, loss = {mse(y_t[i],sg(o))}")

        print('w1 = ',self.w1)
        print('w2 = ',self.w2)


data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

n = net(w1,w2)
n.train(data, all_y_trues, 1000)