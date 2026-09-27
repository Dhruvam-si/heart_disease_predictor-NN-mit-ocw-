import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

keras.utils.set_random_seed(42)
df = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/heart.csv")
print(df.shape)
print(df.head())
print(df.target.value_counts(normalize=True,dropna=False))
categorical_variables = ['sex','cp','fbs','restecg','ca','thal']
numerics = ['age','trestbps','chol','thalach','oldpeak','slope']
df = pd.get_dummies(df,columns=categorical_variables,dtype=float)
test_df = df.sample(frac=0.2,random_state=42)#splitting data for testing
train_df=df.drop(test_df.index)#removing testing data from sample and remaining is training data

means= train_df[numerics].mean() 
sd = train_df[numerics].std()

train_df[numerics]=(train_df[numerics]-means)/sd
test_df[numerics]=(test_df[numerics]-means)/sd

train = train_df.to_numpy()
test = test_df.to_numpy()

train_x = np.delete(train, 7,axis=1)#target column is 7th counting from 0 from left
test_x = np.delete(test,7,axis=1)
print(train_x.shape,test_x.shape) # checking the deletion of one column entirely the one which have answer.

#now from test and train we extract answer keys as it is the 7th column
train_y = train[:, 7]
test_y = test[:, 7]

nums_columns = train_x.shape[1]
#29 as train_x.shape = (242,29) 
#[1]=29 which tells keras 29 features per patient
input = keras.Input(shape=(nums_columns,))
# print(train_df.dtypes)
# print(train.dtype)
# print(train_x.dtype)
# print(train_y.dtype)
# print(train_y[:20])
# print(train_y.min(), train_y.max())
# print(np.unique(train_y))
h = keras.layers.Dense(16,activation="relu",name="hidden")(input)

output = keras.layers.Dense(1,activation="sigmoid",name = "output")(h)
model = keras.Model(input,output)
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
history = model.fit(train_x,train_y,epochs=75,batch_size=32,verbose=1,validation_split=0.2)
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
model.evaluate(train_x,train_y)



