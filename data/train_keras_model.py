from keras.models import Sequential
import pandas

x_train = pandas.DataFrame(pandas.read_pickle('x_train_as_bows.pkl')).as_matrix()
y_train = pandas.DataFrame(pandas.read_pickle('bow/y_train.pkl')).as_matrix()[:1000]

model = Sequential()

from keras.layers import Dense

model.add(Dense(units=1000, activation='relu', input_dim=23735))

for _ in range(20):
    model.add(Dense(units=1000, activation='relu'))

model.add(Dense(units=744, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=50, batch_size=100)
