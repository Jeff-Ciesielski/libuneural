import numpy as np
import sys


def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(int(sys.argv[1])):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    print l0
    print syn0
    print np.dot(l0, syn0)
    print l1
    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)
    print l2_error
    print l2_delta
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    #print "L0:\n",l0
    #print "syn0:\n", syn0
    #print "L1:\n", l1
    #print "L1_error:\n", l1_error
    #print "L1_delta:\n", l1_delta
    #print "syn1:\n", syn1
    #print "L2:\n", l2
    #print "L2_error:\n", l2_error
    #print "L2_delta:\n", l2_delta
