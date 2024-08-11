import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix
import seaborn as sns
import pickle
import math

np.random.seed(42)


num_samples = 500
image_resize_shape_for_lenet = (28,28)
model_path = './model/best_model.pkl'
lr = 0.0003
num_epochs = 30
batch_size = 32

def read_all_image():
    train_a = pd.read_csv('.\\dataset\\NumtaDB_with_aug\\training-a.csv')
    train_b = pd.read_csv('.\\dataset\\NumtaDB_with_aug\\training-b.csv')
    train_c = pd.read_csv('.\\dataset\\NumtaDB_with_aug\\training-c.csv')
    test = pd.read_csv('.\\dataset\\NumtaDB_with_aug\\training-d.csv')

    train = pd.concat([train_a, train_b, train_c], ignore_index=True)

    train = train.sample(frac=1).reset_index(drop=True)
    
    image_matrix = []
    label = []

    i = 0
    

    for index in train.index:
        file_path = '.\\dataset\\NumtaDB_with_aug\\' + train['database name'][index] + "\\" + train['filename'][index]
        img = Image.open(file_path)
        img = img.convert("RGB")
        
        img = img.resize(image_resize_shape_for_lenet)
        img = np.array(img)
        img = (255-img)/255
        img = np.moveaxis(img, -1, 0)
        
        image_matrix.append(img)
        label.append(train['digit'][index])

    test_image_matrix = []
    test_label = []

    i = 0
    
    for index in test.index:
        file_path = '.\\dataset\\NumtaDB_with_aug\\' + test['database name'][index] + "\\" + test['filename'][index]
        img = Image.open(file_path)
        img = img.convert("RGB")
        
        img = img.resize(image_resize_shape_for_lenet)
        img = np.array(img)
        img = (255-img)/255
        img = np.moveaxis(img, -1, 0)
        test_image_matrix.append(img)
        test_label.append(test['digit'][index])

    label = np.array(label)
    t_label = np.array(test_label)


    train_label = np.zeros((label.size, label.max() + 1))
    train_label[np.arange(label.size), label] = 1


    test_label = np.zeros((t_label.size, t_label.max() + 1))
    test_label[np.arange(t_label.size), t_label] = 1

    return np.array(image_matrix), np.array(train_label), np.array(test_image_matrix), np.array(test_label)


image_matrix, label, test_image_matrix, test_label = read_all_image()


def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )

from abc import ABC, abstractmethod
class LayerInstance(ABC):
    @abstractmethod
    def forward(self, input):
        pass
    
    @abstractmethod
    def backward(self, del_v, lr):
        pass

    def update_learnable_parameters(self, del_w, del_b, lr):
        pass

    def get_layer_state(self):
        pass
    

class Conv2D(LayerInstance):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        self.weight = np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size) * math.sqrt(2 / (self.out_channels * self.kernel_size * self.kernel_size))
        self.bias = np.zeros(self.out_channels)


    def __str__(self) -> str:
        return "conv2d"

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        windows = getWindows(x, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride)

        out = np.einsum('bihwkl,oikl->bohw', windows, self.weight)

        out += self.bias[None, :, None, None]

        self.cache = x, windows
        return out
    
    def backward(self, dout,lr=0.001):
        x, windows = self.cache

        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        dout_windows = getWindows(dout, x.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride - 1)
        rot_kern = np.rot90(self.weight, 2, axes=(2, 3))

        db = np.sum(dout, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', windows, dout)
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        self.update_learnable_parameters(dw, db, lr)

        return dx

    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weight -= lr * del_w
        self.bias -= lr * del_b
        # pass
    
    def set_learnable_parameters(self, wt, bs):
        self.weight = wt
        self.bias = bs
        
    def get_layer_state(self):
        return {'name': self.__str__(), 'weights': self.weight, 'bias': self.bias, 'stride': self.stride, 
                'padding': self.padding, 'kernel_size': self.kernel_size,
                'in_channels': self.in_channels, 'out_channels': self.out_channels}

class MaxPoolLayer(LayerInstance):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.mask = None

    def __str__(self) -> str:
        return "maxpool"

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        windows = getWindows(x, (n, c, out_h, out_w), self.kernel_size, stride=self.stride)

        out = np.max(windows, axis=(4, 5))

        self.input = x

        if self.kernel_size == self.stride:
            maxs = out.repeat(self.stride, axis=2).repeat(self.stride, axis=3)
            x_window = x[:, :, :out_h * self.stride, :out_w * self.stride]
            mask = np.equal(x_window, maxs).astype(int)
            self.mask = mask
        
        return out
    
    def backward(self, dout, lr=0.001):
        if self.kernel_size == self.stride:
            mask = self.mask
            dA = dout.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)
            dA = np.multiply(dA, mask)
            pad = np.zeros(self.input.shape)
            pad[:, :, :dA.shape[2], :dA.shape[3]] = dA
            return pad
        
        else:
            dA = np.zeros(self.input.shape)

            for i in range(self.input.shape[0]):
                a = self.input[i] 

                for channel in range(dout.shape[1]):
                    for height in range(dout.shape[2]):
                        for width in range(dout.shape[3]):
                            vert_start = height * self.stride
                            vert_end = vert_start + self.kernel_size
                            horiz_start = width * self.stride
                            horiz_end = horiz_start + self.kernel_size


                            a_slice = a[channel, vert_start: vert_end, horiz_start: horiz_end]
                            mask = a_slice == np.max(a_slice)
                            dA[i, channel, vert_start: vert_end, horiz_start: horiz_end] += dA[i, channel, height, width] * mask
            return dA
    
    def update_learnable_parameters(self, del_w, del_b, lr):
        pass

    def get_layer_state(self):
        return {'name': self.__str__(), 'kernel_size': self.kernel_size, 'stride': self.stride}
    



class FlattenLayer(LayerInstance):
    def __init__(self) -> None:
        self.shape = None

    def __str__(self) -> str:
        return "flatten"
        
    def forward(self, x):
        n,c,h,w = x.shape
        self.shape = x.shape

        out = np.reshape(np.copy(x), (n, c*h*w))
        out = out.T

        return out
    
    def backward(self, dout, lr=0.001):
        dx = np.reshape(np.copy(dout), self.shape)
        return dx

    def update_learnable_parameters(self, del_w, del_b, lr):
        return super().update_learnable_parameters(del_w, del_b, lr)

    def get_layer_state(self):
        return {'name': self.__str__()}


class ReLU(LayerInstance):
    def __init__(self) -> None:
        self.input = None
    
    def __str__(self) -> str:
        return "relu"

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output
    
    def backward(self, dout,lr=0.001):
        input_copy = np.copy(self.input)
        input_copy[input_copy > 0] = 1
        input_copy[input_copy <= 0] = 0
        dx = dout * input_copy
        return dx

    def update_learnable_parameters(self, del_w, del_b, lr):
        return super().update_learnable_parameters(del_w, del_b, lr)

    def get_layer_state(self):
        return {'name': self.__str__()}



class FullyConnectedLayer(LayerInstance):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        self.cache = None

    def __str__(self) -> str:
        return "dense"

    def forward(self, x):
        self.cache = x
        out = np.dot(self.weight, x) + self.bias[:, np.newaxis]
        return out
    
    def backward(self, dout, lr=0.001):
        x = self.cache
        db = np.sum(dout, axis=1)
        dw = np.dot(dout, x.T)
        dx = np.dot(self.weight.T, dout)
        self.update_learnable_parameters(dw, db, lr=lr)
        return dx
    
    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weight -= lr * del_w
        self.bias -= lr * del_b

    def set_learnable_parameters(self, wt,bs):
        self.weight = wt
        self.bias = bs

    def get_layer_state(self):
        return {'name': self.__str__(), 'weight': self.weight, 'bias': self.bias, 'in_features': self.in_features, 'out_features': self.out_features}


class SoftMax(LayerInstance):
    def __init__(self):
        self.input = None

        def __str__(self) -> str:
            return "softmax"

    def forward(self, input):
        self.input = input
        input = input - np.max(input, axis=0)
        exp = np.exp(input)
        output = exp / np.sum(exp, axis=0)
        return output
    
    def backward(self, dout, lr=0.001):
        return np.copy(dout)

    def update_learnable_parameters(self, del_w, del_b, lr):
        return super().update_learnable_parameters(del_w, del_b, lr)

    def get_layer_state(self):
        return {'name': self.__str__()}

class Model():
    def __init__(self):
        self.layers = []
        self.best_valid_f1 = -float("Inf")
    
    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        out = input
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, gradient, learning_rate=0.001):
        back = gradient
        for layer in reversed(self.layers):
            back = layer.backward(back, learning_rate)
        return back

    def fit(self, X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=10, batch_size=32):
        train_loss = []
        train_accruacy = []
        validation_loss = []
        validation_accruacy = []
        validation_f1 = []

        # self.best_valid_f1 = best_valid_f1
        valid_f1 = self.best_valid_f1

        num_batches = int(X_train.shape[0] / batch_size)
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            loss = 0
            accuracy = 0
            for i in tqdm(range(num_batches)):
                X_batch = X_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]
                out = self.forward(X_batch)
                
                loss += log_loss(y_batch, out.T)
                accuracy += accuracy_score(np.argmax(y_batch, axis=1), np.argmax(out.T, axis=1))
                
                gradient = out - y_batch.T
                self.backward(gradient, learning_rate)
            
            train_loss.append(loss/num_batches)
            train_accruacy.append(accuracy/num_batches)

            valid_output = self.forward(X_val)
            valid_loss = log_loss(y_val, valid_output.T)
            valid_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(valid_output.T, axis=1))
            valid_f1 = f1_score(np.argmax(y_val, axis=1), np.argmax(valid_output.T, axis=1), average='macro')
            validation_loss.append(valid_loss)
            validation_accruacy.append(valid_accuracy)
            validation_f1.append(valid_f1)

            if valid_f1 > self.best_valid_f1:
                self.best_valid_f1 = valid_f1
                print("Saving model")
                self.save_state_dict(model_path)

            print(f"Train loss: {train_loss[-1]} | train accuracy: {train_accruacy[-1]} | ")
            print(f"validation loss: {validation_loss[-1]}, validation accuracy: {validation_accruacy[-1]}")
            print(f"Validation F1: {validation_f1[-1]}")
       
        

        return train_loss, train_accruacy, validation_loss, validation_accruacy, validation_f1
        

    def evaluate(self, X_test, y_test, batch_size = 32):
        num_batches = int(X_test.shape[0] / batch_size)
        loss = 0
        accuracy = 0
        for i in tqdm(range(num_batches)):
            X_batch = X_test[i*batch_size:(i+1)*batch_size]
            y_batch = y_test[i*batch_size:(i+1)*batch_size]

            out = self.forward(X_batch)
            loss += log_loss(y_batch, out.T)
            accuracy += accuracy_score(np.argmax(y_batch, axis=1), np.argmax(out.T, axis=1))

        print(f"Loss: {loss/num_batches}, accuracy: {accuracy/num_batches}")
        return loss/num_batches, accuracy/num_batches


    def predict(self, X_test):
        out = self.forward(X_test)
        return np.argmax(out.T, axis=1)


    def save_state_dict(self, file_path):
        state_dict = []
        for layer in self.layers:
            state_dict.append(layer.get_layer_state())
        
        state_dict.append({'name': 'best_valid','value': self.best_valid_f1})

        with open(file_path, 'wb') as f:
            pickle.dump(state_dict, f)
        

    def load_state_dict(self, file_path):
        with open(file_path, 'rb') as f:
            state_dict = pickle.load(f)
        
        for layer_description in state_dict:
            if layer_description['name'] == 'conv2d':
                layer = Conv2D(layer_description['in_channels'], layer_description['out_channels'], layer_description['kernel_size'], layer_description['stride'], layer_description['padding'])
                layer.set_learnable_parameters(layer_description['weights'], layer_description['bias'])
                self.addLayer(layer)

            elif layer_description['name'] == 'maxpool':
                layer = MaxPoolLayer(layer_description['kernel_size'], layer_description['stride'])
                self.addLayer(layer)

            elif layer_description['name'] == 'flatten':
                layer = FlattenLayer()
                self.addLayer(layer)
                
            elif layer_description['name'] == 'relu':
                layer = ReLU()
                self.addLayer(layer)
                
            elif layer_description['name'] == 'dense':
                layer = FullyConnectedLayer(layer_description['in_features'], layer_description['out_features'])
                layer.set_learnable_parameters(layer_description['weight'], layer_description['bias'])
                self.addLayer(layer)

            elif layer_description['name'] == 'softmax':
                layer = SoftMax()
                self.addLayer(layer)
            
            elif layer_description['name'] == 'best_valid':
                self.best_valid_f1 = layer_description['value']
            



def build_lenet5():
    model = Model()
    convolution_layer_1 = Conv2D(3,6,5,1,0)
    relu_layer_1 = ReLU()
    max_pool_layer_1 = MaxPoolLayer(2,2)
    convolution_layer_2 = Conv2D(6,16, 5, 1,0)
    relu_layer_2 = ReLU()
    max_pool_layer_2 = MaxPoolLayer(2,2)
    flattern_layer = FlattenLayer()
    fc_layer_1 = FullyConnectedLayer(256, 120)
    relu_layer_3 = ReLU()
    fc_layer_2 = FullyConnectedLayer(120, 84)
    relu_layer_4 = ReLU()
    fc_layer_3 = FullyConnectedLayer(84, 10)
    soft_max_layer = SoftMax()

    model.addLayer(convolution_layer_1)
    model.addLayer(relu_layer_1)
    model.addLayer(max_pool_layer_1)
    model.addLayer(convolution_layer_2)
    model.addLayer(relu_layer_2)
    model.addLayer(max_pool_layer_2)
    model.addLayer(flattern_layer)
    model.addLayer(fc_layer_1)
    model.addLayer(relu_layer_3)
    model.addLayer(fc_layer_2)
    model.addLayer(relu_layer_4)
    model.addLayer(fc_layer_3)
    model.addLayer(soft_max_layer)

    return model

def train_val_split(X, y, validation_split = 0.2):
    num_samples = X.shape[0]
    num_val_samples = int(num_samples * validation_split)
    X_train = X[num_val_samples:]
    y_train = y[num_val_samples:]
    X_val = X[:num_val_samples]
    y_val = y[:num_val_samples]
    return X_train, y_train, X_val, y_val




def main():
    model = build_lenet5()

    X_train, y_train, X_val, y_val = train_val_split(image_matrix, label, validation_split=0.2)

    train_loss, train_accruacy, validation_loss, validation_accruacy, valid_f1 = model.fit(X_train, y_train, X_val, y_val, learning_rate=lr, epochs=num_epochs, batch_size=batch_size)


    X = range(num_epochs)

    plt.plot(X, train_loss, label='Train loss')
    plt.plot(X, train_accruacy, label='Train accuracy')
    plt.plot(X, validation_loss, label='validation loss')
    plt.plot(X, validation_accruacy, label='validation accuracy')
    plt.plot(X, valid_f1, label='macro f1')

    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()

    plt.savefig('accuracy loss and f1 graph.png')
    plt.show()



    model_1 = Model()
    model_1.load_state_dict(model_path)



    out = model_1.predict(test_image_matrix)
    out_2 = np.copy(out)
    out_2 = out_2.reshape(1, -1)
    y_true = np.argmax(test_label, axis=1)

    y_pred = np.eye(10)[out_2[0]]
    y_true = np.eye(10)[y_true]

    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()


    loss, accuracy = model_1.evaluate(test_image_matrix, test_label)

    epochs_array = [x+1 for x in range(num_epochs)]

    csv_file_matrix = []

    for i in range(len(epochs_array)):
        csv_file_matrix.append([epochs_array[i], train_loss[i], train_accruacy[i], validation_loss[i], validation_accruacy[i], valid_f1[i]])

    file_name = './model/metrics.csv'

    df = pd.DataFrame(csv_file_matrix, columns=['epochs', 'train_loss', 'train_accruacy', 'validation_loss', 'validation_accruacy', 'valid_f1'])
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()
