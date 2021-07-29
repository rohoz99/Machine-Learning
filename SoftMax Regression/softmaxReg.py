import numpy as np

class SoftmaxRegression:

    def __init__(self,learn_rate=1e-5,num_iterations=300000):
        self.learn_rate = learn_rate
        self.num_iterations = num_iterations


#Learning algorithm for fitting the training data
    def learn(self,x,y):

        self.unique_y_vals = np.unique(y)
        self.y_labels = {p: k for k, p in enumerate(self.unique_y_vals)}
        # Inserting X into a numPy array
        x= np.insert(x,0,1,axis=1)
        #Finding the unique elements in Y i.e lager, stout and ale
        # Running Y through one_hot encoding
        y = self.one_hot(y)
        #Weighting scheme matrix creation
        self.weight = np.zeros(shape=(len(self.unique_y_vals),x.shape[1]))
        self.fitting(x,y,learn_rate=1e-5)
        lam = 1



#Calculating and modifying weight for each iteration ran
    def fitting(self,x,y,learn_rate):
        i = 0
        while (not self.num_iterations or i < self.num_iterations):
            index = np.random.choice(x.shape[0], x.shape[0])
            #learn_rate = 1e-5
            xN,yN = x[index],y[index]
            #predictions = self.predictor(xN)
            #Error rate is edited
            #error = y-predictions

            #Error is calculated by the difference of value Y and the predicted value
            error = yN - self.predictor(xN)
            #Gradient formula. The gradient is the dot product of X and the transpose matrix T
            gradient = (learn_rate * np.dot(error.T, xN))
            #Weight is updated
            self.weight += gradient
            #gradient = np.dot(error.T,xN)

#If the absolute value of the maxima of the gradient exceeds the learn rate  then the function breaks
            if np.abs(gradient).max() < self.learn_rate: break


#One-Hot Encoding for the style class (Y)

    def one_hot(self, y):
        return np.eye(len(self.unique_y_vals))[np.vectorize(lambda c: self.y_labels[c])(y).reshape(-1)]

#Calculating the probability
    def predictor(self,x):
          prob = np.dot(x,self.weight.T).reshape(-1, len(self.unique_y_vals))
          return self.softmax_func(prob)


    def softmax_func(self,sc):
        exponent = np.exp(sc - np.max(sc, axis=1).reshape((-1, 1)))
        norm = np.sum(exponent, axis=1).reshape((-1, 1))
        return exponent / norm


    def predict(self,x):
        x = np.insert(x,0,1, axis=1)
        return np.vectorize(lambda i: self.unique_y_vals[i])(np.argmax(self.predictor(x),axis=1))

    def cross_entropy(self,yhat, y):
        return - np.sum(y * np.log(yhat + 1e-6))

    def return_preds(self,x):
        return np.asarray(self.predict(x))


    def accuracy(self,x,y):
        return np.mean(self.predict(x) == y)


