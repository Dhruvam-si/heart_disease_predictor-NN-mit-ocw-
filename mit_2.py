import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

keras.utils.set_random_seed(42)
(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
print(x_train.shape,y_train.shape)
labels=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
#if we write y_train[0] we get label corresponding for t-shirt/top from labels
print(y_train[0])
print(x_train[0])
x_train = x_train/255.0
x_test=x_test/255.0
input = keras.Input(shape =(28,28))
h = keras.layers.Flatten()(input)
h = keras.layers.Dense(256,activation="relu",name="hidden")(h)
h=keras.layers.Dense(256,activation="relu",name="hidden_1")(h)
output = keras.layers.Dense(10,activation="softmax",name="output")(h)
model = keras.Model(input,output)
model.summary()
keras.utils.plot_model(model,show_shapes = True,show_layer_names = True)
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
batch_size=64
epoch = 13
history = model.fit(x_train,y_train,validation_split=0.2,batch_size=batch_size,epochs=epoch)
history_dict = history.history
history_dict.keys()

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

def plot_acc_curves(history):
  plt.clf()
  history_dict = history.history
  acc = history_dict["accuracy"]
  val_acc = history_dict["val_accuracy"]
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, "bo", label="Training acc")
  plt.plot(epochs, val_acc, "b", label="Validation acc")
  plt.title("Training and validation accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()
  
plot_acc_curves(history)
model.evaluate(x_test,y_test)