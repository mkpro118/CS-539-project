from neural_network.layers.dense import Dense
from neural_network.model.sequential import Sequential
from neural_network.model_selection import KFold, train_test_split
from neural_network.metrics import confusion_matrix, accuracy_score, accuracy_by_label
from neural_network.preprocess import OneHotEncoder
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X, y = np.array(iris.data), np.array(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

labels = np.unique(y)
encoder = OneHotEncoder().fit(labels)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

net = Sequential()

net.add(
    Dense(64, input_shape=4, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(3, activation='softmax')
)

net.compile(cost='crossentropy', metrics=['accuracy_score', 'accuracy_by_label'])

kf = KFold(n_splits=5)

for train, validate in kf.split(X_train):
    net.fit(
        X_train[train],
        y_train[train],
        epochs=10,
        validation_data=(X_train[validate], y_train[validate])
    )


predictions = net.predict(X_test, classify=True)
print()
print(confusion_matrix(y_test, predictions))
print()
print(accuracy_score(y_test, predictions))
print()
print(accuracy_by_label(y_test, predictions))
