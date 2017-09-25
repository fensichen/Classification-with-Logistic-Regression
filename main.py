import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# MNIST with a training set of 60,000, a test set of 10,000  
class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.num_examples  = len(label)
        self.x             = input                   # [num_examples, input_dimension]
        self.y             = label
        self.W             = np.random.randn(n_in, n_out) # n_out : number of class, n_in: input dimension
        self.b             = np.zeros((1, n_out))
        self.reg           = 1e-3
        self.step_size     = 1e-0                    # 
    
    def train(self, lr=0.1, input=None, L2_reg=0.00):

        #p_y_given_x        = np.dot(self.x, self.y ) + self.b # np.dot : matrix multiplication here, because input are 2D array  
        #d_y                = self.y - p_y_given_x
        #self.W            += lr* np.dot(self.x, d_y) 
        #self.b            += lr* np.mean(d_y, axis=0)
        loss, scores, probs = self.negative_log_likelihood() 
        print "loss", loss
        dscores             = probs
        dscores[range(self.num_examples), self.y] -= 1
        dscores             = dscores/self.num_examples
        dW                  = np.dot(self.x.transpose(), dscores)
        db                  = np.sum(dscores, axis=0, keepdims=True) # summation over all rows for each column 
        dW                 += self.reg*self.W

        self.W             += -self.step_size * dW
        self.b             += -self.step_size * db
        return loss

    def softmax(self, x):

        # get un-normalized probabilities
        exp_scores         = np.exp(x)
        #print "exp_scores"
        
        # normalize them for each sample
        if exp_scores.ndim == 1:
            probs = exp_scores/np.sum(exp_scores, axis=0)
        else:
            a = np.sum(exp_scores, axis=1) 
            a = a[:, np.newaxis]
            probs = exp_scores/a # [ NxK]

        print "probs", probs
        
        b = np.where(probs == 0)[0]
        #rint "b.shape", b.shape
        return probs

    def negative_log_likelihood(self):
        
        # compute class scores for a liner classifier
        scores             = np.dot(self.x, self.W) + self.b # [ number_of_points, n_out(class_number) ]
        print "scores", scores
        probs              = self.softmax( scores )               # [ number_of_points, n_out(class_number) ]
        vec =  probs[range(self.num_examples), self.y] 
        print "vec", vec
        correct_logprobs   = -1*np.log( probs[range(self.num_examples), self.y] )
        print "correct_logprobs", correct_logprobs
        data_loss          = np.sum(correct_logprobs)/self.num_examples
        print "data_loss", data_loss
        return data_loss, scores, probs


digits = load_digits()
print digits.data.shape
print digits.target.shape

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1,5, index + 1)
    plt.imshow( np.reshape(image, (8,8)), cmap=plt.cm.gray)
#plt.show()
    #plt.title('Training: %i\n', label, fontsize=20)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
print "x_train.shape", x_train.shape, "x_test.shape", x_test.shape
print "y_train.shape", y_train.shape, "y_test.shape", y_test.shape

input_dim  = 8*8
output_dim = 10
classifier = LogisticRegression(input=x_train, label=y_train, n_in=input_dim, n_out=output_dim)

n_epochs   = 3
for epoch in np.arange(0,n_epochs):
    loss = classifier.train(lr=10-3)
    
    print "epoch: ", epoch, " loss: ", loss
