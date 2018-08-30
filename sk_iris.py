from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


"""Arbitrary Data Loading"""
# For feature scaling
sc = StandardScaler()

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Split 30% test, 70% train, random seed is 1.
# We stratify to keep data proportionate
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1,
                                                    stratify=y)

# Esitmate the mean and the standard deviation of each feature
sc.fit(X_train)

# Standardize the inputs via standardization
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # Marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # Plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test Set')


def sk_perceptron():
    # For feature scaling
    sc = StandardScaler()

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Split 30% test, 70% train, random seed is 1.
    # We stratify to keep data proportionate
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)

    # Esitmate the mean and the standard deviation of each feature
    sc.fit(X_train)

    # Standardize the inputs via standardization
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Init a new perceptron 40 epochs and an alpha of 0.1
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: {}'.format((y_test != y_pred).sum()))

    print('Accuracy of results: {}'.format(accuracy_score(y_test, y_pred)))
    print('Test accuracy: {}'.format(ppn.score(X_test_std, y_test)))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined_std,
                          y_combined,
                          ppn,
                          range(105, 150))
    plt.xlabel('Petal Length [Standardized]')
    plt.ylabel('Petal Width [Standardized]')
    plt.legend(loc='upper left')
    plt.show()


def sk_logistic_regression():
    # For feature scaling
    sc = StandardScaler()

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Split 30% test, 70% train, random seed is 1.
    # We stratify to keep data proportionate
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)

    # Esitmate the mean and the standard deviation of each feature
    sc.fit(X_train)

    # Standardize the inputs via standardization
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    lr = LogisticRegression(C=100.0, random_state=1)

    lr.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std,
                          y_combined,
                          lr,
                          range(105, 150))
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.show()


def sk_svm():
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    # For feature scaling
    sc = StandardScaler()

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Split 30% test, 70% train, random seed is 1.
    # We stratify to keep data proportionate
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)

    # Esitmate the mean and the standard deviation of each feature
    sc.fit(X_train)

    # Standardize the inputs via standardization
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std,
                          y_combined,
                          svm,
                          range(105, 150))
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.show()


def sk_decision_tree():
    """This sample overifts for depth > 5"""
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=3,
                                  random_state=1)
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    plot_decision_regions(X_combined,
                          y_combined,
                          classifier=tree,
                          test_idx=range(105, 150))
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend(loc='upper left')
    plt.show()


def sk_random_forest_classifier():
    """
    n_esitmators -- The number of decision trees,
    criterion -- The impurity classifier
    n_jobs -- Parallelize the model training (for larger sets)
    corresponds to # of cores on system
    """
    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=25,
                                    random_state=1,
                                    n_jobs=6)
    forest.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    plot_decision_regions(X_combined,
                          y_combined,
                          classifier=forest,
                          test_idx=range(105, 150))
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend(loc='upper left')
    plt.show()


def sk_k_nearest_neighbors():
    knn = KNeighborsClassifier(n_neighbors=5,
                               p=2,
                               metric='minkowski')
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std,
                          y_combined,
                          classifier=knn,
                          test_idx=range(105, 150))
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend(loc='upper left')
    plt.show()


# sk_logistic_regression()
sk_svm()
# sk_decision_tree()
# sk_random_forest_classifier()
# sk_k_nearest_neighbors()
