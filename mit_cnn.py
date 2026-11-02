import tensorflow 
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

keras.utils.set_random_seed(42)

labels = ["T-shirt/top",
          "Trouser",
          "Pullover",
          "Dress",
          "Coat",
          "Sandal",
          "Shirt",
          "Sneaker",
          "Bag",
          "Ankle boot"]

def plot_loss_curves(history):
  plt.clf()
  history_dict = history.history
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

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train/ 255.0
x_test = x_test/ 255.0

#entering convulational territories of keras and tensorflow so we need to expand
#channel dimension which CNN expects it  expects 3 colors usually we only have on b & w
#so we add one more dimension as color channel
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)
#start of first convulational block
input = keras.Input(shape=x_train.shape[1:]) #gives info of dimension and channel per example and not no. of example too
x = keras.layers.RandomZoom(0.2)(input)

x = keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu",padding="same",name="conv_1")(x)
#kernel means size of filter,32 is number of filters,activation relu and name conv_1
x = keras.layers.MaxPool2D()(x) #this is a pooling layer

#this ends first convulational block


x = keras.layers.Conv2D(64,                    # Number of filters
                        kernel_size=(3, 3),    # The shape of each filter
                        activation="relu",padding="same",     # RELU activation as usual
                        name="Conv_2")(x)
x=keras.layers.MaxPool2D()(x)
#this is 2nd cnn block

# x = keras.layers.Conv2D(32,kernel_size=(4, 4),activation="relu",padding="same",
#                         name="Conv_3")(x)
# x=keras.layers.MaxPool2D()(x)
#end of 3rd cnn block
# x = keras.layers.Conv2D(64,kernel_size=(4, 4),activation="relu",padding="same",
#                         name="Conv_4")(x)
# x=keras.layers.MaxPool2D()(x)
#end of 4th cnn block
# x = keras.layers.Conv2D(32,kernel_size=(4, 4),activation="relu",padding="same",
#                         name="Conv_5")(x)
# x=keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.25)(x)
#end of 5th cnn block
# x = keras.layers.Conv2D(32,kernel_size=(4, 4),activation="relu",padding="same",
#                         name="Conv_4")(x)
# x=keras.layers.MaxPool2D()(x)
x = keras.layers.Flatten()(x)
#FLATTENING OF LAYER
x = keras.layers.Dense(64,activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)  
x = keras.layers.Dense(128,activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)  
x = keras.layers.Dense(128,activation="relu")(x)




output = keras.layers.Dense(10,activation="softmax")(x)
model = keras.Model(input,output)
model.summary()
keras.utils.plot_model(model, show_shapes=True)

lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    mode="max",        # higher is better for accuracy
    factor=0.5,
    patience=2,
    min_delta=0.001,   # needs at least +0.1% to count as improvement
    min_lr=1e-5,
)

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=30,
                    validation_split=0.2,
                    callbacks=[lr_cb])
plot_loss_curves(history)
plot_acc_curves(history)
score = model.evaluate(x_test, y_test)
print("Test accuracy:", score[1])
#-----now optimizing------
# getting predcitions of hat
y_hat = np.argmax(model.predict(x_test), axis=1)
print(model.predict(x_test).shape)
# #axis one cause we have 2 dimension matrix for data

#  collect examples where the model made a mistake
misses = np.where(y_hat != y_test)
#now we create a confusion matrix to understand where the model 
#fumbles.
actuals = [labels[i] for i in y_test]
predictions = [labels[i] for i in y_hat]

df = pd.DataFrame({'predictions':predictions, 'actuals':actuals})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

print(pd.crosstab(df.predictions, df.actuals))
