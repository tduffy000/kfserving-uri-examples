from sklearn import svm
from sklearn import datasets
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def train(X, y):
    clf = svm.SVC(gamma='auto')
    clf.fit(X, y)
    return clf

def convert_to_onnx(clf):
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    return onx

def freeze(onx, path='../frozen'):
    with open(f'{path}/iris.onnx', 'wb') as out:
        out.write(onx.SerializeToString())
    return True

if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = train(X, y)
    onx = convert_to_onnx(clf)
    freeze(onx)