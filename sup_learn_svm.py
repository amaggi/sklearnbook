import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics


faces = fetch_olivetti_faces()

# learn about the data
print faces.DESCR
print faces.keys()
print faces.images.shape
print faces.data.shape

# check data normalization
print np.max(faces.data)
print np.min(faces.data)
print np.mean(faces.data)

def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05,
                        wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20 x 20
        p = fig.add_subplot(20, 20, i+1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the limage with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))

    fig.savefig('faces.png')

print_faces(faces.images, faces.target, 20)

# linear SVC
svc_1 = SVC(kernel='linear')
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target,
                                                    test_size=0.25,
                                                    random_state=0)

def mean_score(scores):
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores),
                                                       sem(scores))

def evaluate_cross_validation(clf, X, y, K):
    # create a k-foold cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # default score (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print mean_score(scores)

evaluate_cross_validation(svc_1, X_train, y_train, 5)

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print "Accuracy on training set:"
    print clf.score(X_train, y_train)
    print "Accuracy on testing set:"
    print clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)

    print "Classification Report:"
    print metrics.classification_report(y_test, y_pred)
    print "Confusion Matrix:"
    print metrics.confusion_matrix(y_test, y_pred)

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
