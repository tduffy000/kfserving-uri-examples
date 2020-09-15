## KFServing URI Storage 
The [kfserving](https://github.com/kubeflow/kfserving) project provides a serverless framework to run many Machine Learning model frameworks on Kubernetes with minimal effort from data scientists. 

In order to allow for easier testing, learning, and proof-of-concept (PoC) work, the ability to define a generic URI was added in this [PR](https://github.com/kubeflow/kfserving/pull/979). This repo provides examples of defining (very minimal) models  in various frameworks and their frozen representations, along with the Custom Resource Definition (CRD) to apply them to an existing Kubernetes cluster with kfserving running.

### Example
The easiest model example is likely that of [Scikit-Learn](https://scikit-learn.org/stable/index.html) which most students and professionals are familiar with. In [`./sklearn/model/train.py`](./sklearn/model/train.py), we specify a sample Iris dataset model to predict flowers, 
```python
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
```

You can use the frozen model object in [`./sklearn/frozen/model.joblib`](`./sklearn/frozen/model.joblib`) to test out deploying this model in kfserving. 

The CRD for that would look like this:
```yaml
apiVersion: serving.kubeflow.org/v1alpha2
kind: InferenceService
metadata:
  name: sklearn-from-uri
spec:
  default:
    predictor:
      sklearn:
        storageUri: https://github.com/tduffy000/kfserving-uri-examples/blob/master/sklearn/frozen/model.joblib?raw=true
```
The full example can be found [here](https://github.com/kubeflow/kfserving/tree/master/docs/samples/uri).