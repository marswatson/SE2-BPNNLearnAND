import math
import random
import numpy as np
from numpy import random

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

class BPNN:
    def __init__(self,Ni, Nh, No):
        #set the neural network have 3 layers
        self.Ni = Ni
        self.Nh = Nh
        self.No = No

        #set the weight matrix
        self.Wi = 2*random.rand(Ni,Nh) - 1
        self.Wo = 2*random.rand(Nh,No) - 1
        #self.Wi = np.zeros((Ni,Nh))
        #self.Wo = np.zeros((Nh,No))
        #self.Wi = np.array([[0.02,0.02,0.02],[0.02,0.02,0.02]])
        #self.Wo = np.array([[-1.5],[-1.5],[-1.5]])

        #activate every node vector in neural network
        self.Ai = []
        self.Ah = []
        self.Ao = []

    #forward process
    def FP(self,input):
        #activate the input level
        self.Ai = np.array(input)

        #calculate the output of hidden layer
        self.Ah = np.dot(self.Ai, self.Wi)
        self.Ah = sigmoid(self.Ah)

        #calculate the output of out put layer
        self.Ah = np.array(self.Ah)
        self.Ao = np.dot(self.Ah, self.Wo)
        self.Ao = sigmoid(self.Ao)
        self.Ao = np.array(self.Ao)

        return self.Ao

    #BP process
    def BP(self,target,rate):
        #calculate the delta for output weight change
        output_deltas = [0.0] * self.No
        for k in range(self.No):
            error = target[k]-self.Ao[k]
            output_deltas[k] = dsigmoid(self.Ao[k]) * error

        #update the output weight
        for j in range(self.Nh):
            for k in range(self.No):
                changes = output_deltas[k] * self.Ah[j]
                self.Wo[j][k] += rate*changes

        #calculate the delta for input weight change
        hidden_deltas = [0.0] * self.Nh
        for j in range(self.Nh):
            error = 0.0
            for k in range(self.No):
                error = error + output_deltas[k]*self.Wo[j][k]
            hidden_deltas[j] = dsigmoid(self.Ah[j]) * error

         #update the hidden layer weight
        for j in range(self.Ni):
            for k in range(self.Nh):
                changes = hidden_deltas[k] * self.Ai[j]
                self.Wi[j][k] += rate*changes

        # return error between output and target
        error = 0.0
        for k in range(len(target)):
            error += (target[k] - self.Ao[k])**2
        return math.sqrt(error)

    def train(self, set, rate, target_error):
        #calculate the first batch error

        first =  set [random.randint(0,3)]
        print'Initial hidden layer weight is '
        print self.Wi
        print'Initial output layer weight is '
        print self.Wo
        self.FP(first[0])
        error = self.BP(first[1],rate)
        print'first batch error is %.5f' %error


        #train dataset until error is smaller than the target_error
        average_error = 1.0
        count = 1
        while (average_error > target_error):
            average_error = 0.0
            for p in set:
                self.FP(p[0])
                average_error += self.BP(p[1],rate)
                count += 1
            average_error = average_error/len(set)
            if count >= 1000000:
                break

        print'Final hidden layer weight is '
        print self.Wi
        print'Final output layer weight is '
        print self.Wo
        print 'the final error is %.5f' %average_error
        print 'the total number of batches run through in the training is %d' %count

    def test(self,set):
        for p in set:
            output = self.FP(p[0])
            result = 0
            if output>0.5:
                result = 1
            print p[0],'->',output ,'predict result: ', result

if __name__ == '__main__':
    set = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]]
    ]

    print '---------------------learning rate = 0.5, target error = 0.1---------------------------------'
    bpnn1 = BPNN(2,3,1)
    bpnn1.train(set,0.5,0.1)
    bpnn1.test(set)
    print '---------------------------------------------------------------------------------------------'

    print '---------------------learning rate = 1, target error = 0.1-----------------------------------'
    bpnn2 = BPNN(2,3,1)
    bpnn2.train(set,1,0.1)
    bpnn2.test(set)
    print '---------------------------------------------------------------------------------------------'

    print '---------------------learning rate = 0.5, target error = 0.02--------------------------------'
    bpnn3 = BPNN(2,3,1)
    bpnn3.train(set,0.5,0.02)
    bpnn3.test(set)
    print '---------------------------------------------------------------------------------------------'


    print '---------------------learning rate = 1, target error = 0.02----------------------------------'
    bpnn4 = BPNN(2,3,1)
    bpnn4.train(set,1,0.02)
    bpnn4.test(set)
    print '---------------------------------------------------------------------------------------------'









