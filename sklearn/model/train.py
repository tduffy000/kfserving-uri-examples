from sklearn import svm
from sklearn import datasets
import joblib

def train(X, y):
    clf = svm.SVC(gamma='auto')
    clf.fit(X, y)
    return clf

def freeze(clf, path='../frozen'):
    joblib.dump(clf, f'{path}/model.joblib')
    return True

if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = train(X, y)
    freeze(clf)