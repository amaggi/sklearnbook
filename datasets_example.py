import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import sem

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

print X_iris.shape, y_iris.shape
print X_iris[0], y_iris[0]
print iris.target_names

# get dataset with only the first two attributes
X, y = X_iris[:, :2], y_iris

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=33)
print X_train.shape, y_train.shape

# stadardize the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# plot training instances in 2D
#colors = ['red', 'greenyellow', 'blue']
#for i in xrange(len(colors)):
#    xs = X_train[:, 0][y_train == i]
#    ys = X_train[:, 1][y_train == i]
#    plt.scatter(xs, ys, c=colors[i])
#plt.legend(iris.target_names)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#plt.savefig('datasets_example_01.png')

# do the fitting on the training set
clf = SGDClassifier()
clf.fit(X_train, y_train)
print clf.coef_
print clf.intercept_

# plot the output
#x_min = X_train[:, 0].min() - .5
#x_max = X_train[:, 0].max() + .5
#y_min = X_train[:, 1].min() - .5
#y_max = X_train[:, 1].max() + .5
#xs = np.arange(x_min, x_max, 0.5)
#fig, axes = plt.subplots(1, 3)
#fig.set_size_inches(10, 6)
#for i in [0, 1, 2]:
#    axes[i].set_aspect('equal')
#    axes[i].set_title('Class ' + str(i) + ' versus the rest')
#    axes[i].set_xlabel('Sepal length')
#    axes[i].set_ylabel('Sepal width')
#    axes[i].set_xlim(x_min, x_max)
#    axes[i].set_ylim(y_min, y_max)
#    plt.sca(axes[i])
#    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
#    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
#    plt.plot(xs, ys, hold=True)
#plt.savefig('datasets_example_02.png')

# make some predictions
print clf.predict(scaler.transform([[4.7, 3.1]]))
print clf.decision_function(scaler.transform([[4.7, 3.1]]))

# evaluate performance
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)
print metrics.accuracy_score(y_train, y_train_pred)
print metrics.accuracy_score(y_pred, y_test)
print metrics.classification_report(y_test, y_pred,
                                    target_names=iris.target_names)
print metrics.confusion_matrix(y_test, y_pred)

# do a k-fold cross-validation
# create a pipeline
clf = Pipeline([('scaler', StandardScaler()),
                ('linear_model', SGDClassifier())])
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
# by default the score used is the one returned by score method of the
# estimator (accuracy)
scores = cross_val_score(clf, X, y, cv=cv)

def mean_score(scores):
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores),
                                                       sem(scores))

print mean_score(scores)
