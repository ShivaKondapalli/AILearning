from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


digits = datasets.load_digits()


def plot_image(n):
    """get random image from the digits data and plot it

    parameter:

    n : int. single image from the images array
    .
    """

    if n > digits.images.shape[0]:
        print('Index out of bounds error, please enter a number lesser than or '
              'equal to what you just entered')

    else:

        plt.figure()
        plt.imshow(digits.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()


# plot_image(200)

X = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train, y_train)

print(knn.score(x_test, y_test))

# Model-Complexity

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)


# plt.figure()
# plt.title('Accuracy v number of neighbors(K)')
# plt.plot(neighbors, train_accuracy, label='Trainig accuracy')
# plt.plot(neighbors, test_accuracy, label='Testing accuracy')
# plt.legend()
# plt.xlabel('Numbe of Neigbhors')
# plt.ylabel('Accuracy')
# plt.show()

array = np.random.random([2, 3])
print(array)


print(array.reshape(-1, 6))  # the first dimension is 6
