### Importing all Nescessary Libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix ,  accuracy_score

class Network:

    def __init__(self,x,y, epochs, nos_of_batches):
        np.random.seed(2)
        self.input = x
        self.output = y
        neuron1 = 32                                            # neurons for hidden layers
        neuron2 = 64
        ip_dim = x.shape[1]                                     # input layer size 64
        op_dim = y.shape[1]                                     # output layer size 10
        self.lr = 0.01                                          # user defined learning rate
        self.samples = x.shape[0]
        self.w1 = np.random.randn(ip_dim, neuron1)              # weights
        self.b1 = np.zeros((1, neuron1))                        # biases
        self.w2 = np.random.randn(neuron1, neuron2) 
        self.b2 = np.zeros((1, neuron2))
        self.w3 = np.random.randn(neuron2, op_dim) 
        self.b3 = np.zeros((1, op_dim))
        self.K = np.random.randn(3,1)                           # Coefficient k0, k1, k2
        self.epochs = epochs                                    # Nos. of Epochs
        self.batch_size = int(x.shape[0]/nos_of_batches)        # Batch Size
        self.nbatches = nos_of_batches                          # Nos. of Batches
        self.trainLossList = []                                 # Train Loss Values
        self.valLossList = []                                   # Validation Loss Values
    
    def softmax(self,s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def avg_cool(self,x):
        return x.mean(0)

    def avg_e(self,x):
        return x.mean()

    def ada_act(self,x):
        return self.K[0] + self.K[1]*x

    def deriv_ada_act(self,x):
        return self.K[1]

    def feedforward(self, a0):
        self.z1 = np.dot(a0, self.w1) + self.b1
        self.a1 = self.ada_act(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.ada_act(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.softmax(self.z3)

    def error(self, pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def backprop(self, a0, output):
        dz3 = self.a3 - output
        dw3 = np.matmul((self.a2.T)/self.batch_size, dz3)
        db3 = self.avg_cool(dz3)
        da2 = np.matmul(dz3, self.w3.T)
        dz2 = self.deriv_ada_act(self.z2)*da2
        dw2 = np.matmul((self.a1.T)/self.batch_size, dz2)
        db2 = self.avg_cool(dz2)
        dK2 = np.array([self.avg_e(da2), self.avg_e(da2 * self.z2), self.avg_e(da2 * (self.z2**2))]).reshape(3,1)
        da1 = np.matmul(dz2, self.w2.T)
        dz1 = self.deriv_ada_act(self.z1)*da1
        dw1 = np.matmul((a0.T)/self.batch_size, dz1)
        db1 = self.avg_cool(dz1)
        dK1 = np.array([self.avg_e(da1), self.avg_e(da1 * self.z1), self.avg_e(da1 * (self.z1**2))]).reshape(3,1)
        dK = dK2 + dK1

        ### Updating Parameters
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.K -=self.lr * dK

    def predict(self, data):
        self.feedforward(data)
        return self.a3  

    def val_predict(self, x_val, y_val):
        a3 = self.predict(x_val)
        val_err = self.error(a3, y_val)
        print("Test Error: ", val_err)
        self.valLossList.append(val_err)

    def train(self,  x_val,y_val):
        for i in range(self.epochs):
            for j in range(self.nbatches):
                a0 = self.input[j:(j+1)*self.batch_size]
                output = self.output[j:(j+1)*self.batch_size]
                self.feedforward(a0)
                self.backprop(a0, output)
            ### Calculating Training Loss after every Epochs
            loss = self.error(self.a3, output)
            self.trainLossList.append(loss)
            print('Train Error: ', loss)
            ### Calculating Validation Loss after every Epochs
            self.val_predict(x_val, np.array(y_val))

        ### Saving Final Parameter to "modelWeight.npy"      
        with open('modelWeights.npy', 'wb') as f:
            parameter = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.K]
            for i in parameter:
                np.save(f, i)
    
    def get_accuracy(self, x, y):
        acc = 0
        for xx,yy in zip(x, y):
            s = self.predict(xx)
            # print(s)
            if s.argmax() == np.argmax(yy):
                acc +=1
        return acc/len(x)*100

def plotgraph(data,xlabel, ylabel, title, filename, ConfusionMatrix):
    
    if ConfusionMatrix:
        plt.imshow(data, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
    else:
        plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def get_metrics(model,x, y):
    y_pred = [model.predict(xx).argmax() for xx in x]
    y_true = [yy.argmax() for yy in y]
    # print("True Labels: ", y_true)
    # print("Predicted Labels: ", y_pred)
    return y_pred, y_true
    
### Main Function

def main():
    data = load_iris()
    X = data.data                                       # Data 
    Y = data.target                                     # Target Values
    new_Y = pd.get_dummies(Y)                           # One Hot Encoding for Target Values

    ### Normalize Train Data
    # xmin, xmax = np.min(X), np.max(X)
    # new_X = (X - xmin)/(xmax- xmin)

    x_train, x_val, y_train, y_val = train_test_split(X, new_Y, test_size=0.25, random_state=20, stratify = None)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, "\n")

    Nos_of_epochs = 100  
    Nos_of_batch = 10

    model = Network(x_train,np.array(y_train), Nos_of_epochs, Nos_of_batch)
    model.train(x_val,y_val)

    y_pred, y_true = get_metrics(model, x_val, np.array(y_val))

    print("Training accuracy : ", model.get_accuracy(x_train, np.array(y_train)), "\n")
    print("Test accuracy : ", model.get_accuracy(x_val, np.array(y_val)), "\n")
    print("Classification Report: \n" , classification_report(y_true, y_pred, target_names=['class0', 'class1', 'class2']), "\n")
    print("Accuarcy Score: \n", accuracy_score(y_true, y_pred), "\n")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    print("Confusion Matrix: \n", cm)
    

    plotgraph(cm, "Predicted Labels", "True Labels", "Confusion Matrix", "ConfusionMatrix.jpg", True)
    plotgraph(model.valLossList,"Epochs", "Loss Value", "Validation Loss Vs Epochs", "ValLoss.jpg", False)
    plotgraph(model.trainLossList,"Epochs", "Loss Value", "Train Loss Vs Epochs", "TrainLoss.jpg", False)

if __name__ == "__main__":
    main()