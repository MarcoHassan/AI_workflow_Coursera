# CASE STUDY - unsupervised learning



```python
import os
import joblib
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.svm import SVC

import imblearn.pipeline as pl
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, SVMSMOTE
    
plt.style.use('seaborn')
%matplotlib inline
```

## Synopsis

  > We are now going to predict customer retention.  There are many
    models and many transforms to consider.  Use your knowledge of
    pipelines and functions to ensure that your code makes it easy to
    compare and iterate.
    
  > Marketing has asked you to make a report on customer retention.
    They would like you to come up with information that can be used
    to improve current marketing strategy efforts.  The current plan
    is for marketing at AAVAIL to collect more features on subscribers
    the and they would like to use your report as a proof-of-concept
    in order to get buyin for this effort.
  
## Outline

1. Create a churn prediction baseline model
2. Use clustering as part of your prediction pipeline
3. Run and experiment to see if re-sampling techniques improve your model

## Data

Here we load the data as we have already done.

`aavail-target.csv`


```python
df = pd.read_csv(os.path.join(".",r"aavail-target.csv"))
```


```python
df.columns
```




    Index(['customer_id', 'is_subscriber', 'country', 'age', 'customer_name', 'subscriber_type', 'num_streams'], dtype='object')




```python
_y = df.pop('is_subscriber') 
```


```python
df.columns
```




    Index(['customer_id', 'country', 'age', 'customer_name', 'subscriber_type', 'num_streams'], dtype='object')



You see that the pop method removes the column from the dataframe in a
similar spirit to the pop method applied to a a list.


```python
## pull out the target and remove uneeded columns
y = np.zeros(_y.size)
y[_y==0] = 1 
df.drop(columns=['customer_id','customer_name'],inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>age</th>
      <th>subscriber_type</th>
      <th>num_streams</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>united_states</td>
      <td>21</td>
      <td>aavail_premium</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>singapore</td>
      <td>30</td>
      <td>aavail_unlimited</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>united_states</td>
      <td>21</td>
      <td>aavail_premium</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>united_states</td>
      <td>20</td>
      <td>aavail_basic</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>singapore</td>
      <td>21</td>
      <td>aavail_premium</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



### QUESTION 1

Create a stratified train test split of the data


```python
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2,
                                                    stratify = y,
                                                    random_state = 1)
```

### QUESTION 2

Create a baseline model.


We are going to test whether clustering followed by a model improves
the results.  The we will test whether re-sampling techniques provide
improvements.  Use a pipeline or another method, but create a baseline
model given the data. Here is the ColumnTransformer we have used
before.

Look at the empty values in the dataset




```python
list(df.describe().columns)
```




    ['age', 'num_streams']




```python
## missing values summary
print("Missing Value Summary\n{}".format("-"*35))
print(df.isnull().sum(axis = 0))
```

    Missing Value Summary
    -----------------------------------
    country            0
    age                0
    subscriber_type    0
    num_streams        0
    dtype: int64



```python
## preprocessing pipeline
numeric_features = ['age', 'num_streams']

# notice that in this script you add a simpleimputer despite the fact
# that no values are missing. This is a sensible solution as you might
# want your slution to stay strong even in production when the data
# that you might gather might be empty or in general when null values
# might well easily occur.

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ['country', 'subscriber_type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
```

# SVC Baseline


```python
## Baseline
param_grid_svm = {
    'svm__C': [0.01,0.1,0.5,1.0,1.5,5.0,10.0],
    'svm__gamma': [0.001,0.01,0.1]
}

best_params = {}
pipe_svm = Pipeline(steps=[('pre', preprocessor),
                           ('svm',SVC(kernel='rbf',
                                      class_weight='balanced'))])

# svc_fit = pipe_svm.fit(X_train, y_train)

# y_pred = svc_fit.predict(X_test)

grid = GridSearchCV(pipe_svm,
                    param_grid=param_grid_svm,
                    cv=5)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

best_params = grid.best_params_

print("Fitted Pipeline\n" + "-->".join(pipe_svm.named_steps.keys()) + "\n{}\n".format("="*35))
print("f1_score",round(f1_score(y_test, y_pred,average='binary'),3))


```

    Fitted Pipeline
    pre-->svm
    ===================================
    
    f1_score 0.634


Random Forest Baseline


```python
param_grid_rf = {
    'rf__n_estimators': [20,50,100,150],
    'rf__max_depth': [4, 5, 6, 7, 8],
    'rf__criterion': ['gini', 'entropy']
}

pipe_rf = Pipeline(steps=[('pre', preprocessor),
                          ('rf',RandomForestClassifier())])

grid = GridSearchCV(pipe_rf,
                    param_grid=param_grid_rf,
                    cv=5)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print("-->".join(pipe_rf.named_steps.keys()))
best_params = dict(best_params, **grid.best_params_)
print("f1_score",round(f1_score(y_test, y_pred,average='binary'),3))

```

    pre-->rf
    f1_score 0.557


### QUESTION 3

The next part is to create version of the classifier that uses
identified clusters.

Here is a class to get you started.  It is a transformer like those
that we have been working with.  There is an example of how to use it
just below.  In this example 4 clusters were specified and their
one-hot encoded versions were appended to the feature matrix.  Now
using pipelines and/or functions compare the performance using cluster
profiling as part of your matrix to the baseline.  You may compare
multiple models and multiple clustering algorithms here.


```python
class KmeansTransformer(BaseEstimator, TransformerMixin): ## inheriting
                                                          ## baseestimator
                                                          ## and
                                                          ## tranformermixin
    def __init__(self, k=4):
        self.km = KMeans(n_clusters=k,n_init=20)
        
    def transform(self, X, *_):
        labels = self.km.predict(X)
        enc = OneHotEncoder(categories='auto') # here you determine
                                               # categories to
                                               # hotencode
                                               # automatically.
        oh_labels = enc.fit_transform(labels.reshape(-1,1))
        oh_labels = oh_labels.todense() ## converts to matrix
        return(np.hstack((X,oh_labels)))

    def fit(self,X,y=None,*_):
        self.km.fit(X)
        labels = self.km.predict(X)
        self.silhouette_score = round(silhouette_score(X,labels,metric='mahalanobis'),3)
        return(self)

class GmmTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=4):
        self.gmm = BayesianGaussianMixture(n_components=k,
                                           covariance_type='full',
                                           max_iter=500,
                                           n_init=10,
                                           warm_start=True)        
    def transform(self, X,*_):
        probs = self.gmm.predict_proba(X) + np.finfo(float).eps
        return(np.hstack((X,probs)))
        
    def fit(self,X,y=None,*_):
        self.gmm.fit(X)
        labels = self.gmm.predict(X)
        self.silhouette_score = round(silhouette_score(X,
                                                       labels,
                                                       metric='mahalanobis'),
                                      3)
        return(self)

```


```python
## example for GMM  
preprocessor.fit(X_train)
X_train_pre = preprocessor.transform(X_train)    
gt = GmmTransformer(4)
gt.fit(X_train_pre)
X_train_gmm = gt.transform(X_train_pre)

print("Check how the feature space is increased by the cluster dummy obtained via the clustering algo.\n{}\n".format("-"*35))
print(X_train_pre.shape); print(X_train_gmm.shape)

## example for kmeans
preprocessor.fit(X_train)
X_train_pre = preprocessor.transform(X_train)    
kt = KmeansTransformer(4)
kt.fit(X_train_pre)
X_train_kmeans = kt.transform(X_train_pre)

print("\n{}\n{}".format(X_train_pre.shape, X_train_kmeans.shape))
```

    Check how the feature space is increased by the cluster dummy obtained via the clustering algo.
    -----------------------------------
    
    (800, 7)
    (800, 11)
    
    (800, 7)
    (800, 11)



```python
def run_clustering_pipeline(smodel,umodel):
    fscores,sscores = [],[]
    for n_clusters in np.arange(3,8):
        
        if smodel == 'rf':
            clf = RandomForestClassifier(n_estimators=best_params['rf__n_estimators'],
                                         max_depth=best_params['rf__max_depth'],
                                         criterion=best_params['rf__criterion'])
        elif smodel == 'svm':
            clf = SVC(C=best_params["svm__C"],
                      gamma=best_params["svm__gamma"])
        else:
            raise Exception("invalid supervised learning model")
        
        if umodel == 'gmm':
            cluster = GmmTransformer(n_clusters)
            
        elif umodel == 'kmeans':
            cluster = KmeansTransformer(n_clusters)
            
        else:
            raise Exception("invalid unsupervised learning model")
        
        pipe = Pipeline(steps=[('pre', preprocessor),
                               ('clustering', cluster),
                               ('classifier', clf)])  
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score = round(f1_score(y_test, y_pred, average='binary'),3)
        fscores.append(score)
        sscores.append(pipe['clustering'].silhouette_score)
        
    return(fscores)

```


```python
## run the different iteration of the model
cp_results = {}
cp_results['svm-kmeans'] = run_clustering_pipeline('svm','kmeans')
cp_results['svm-gmm'] = run_clustering_pipeline('svm','gmm')
cp_results['rf-kmeans'] = run_clustering_pipeline('rf','kmeans')
cp_results['rf-gmm'] = run_clustering_pipeline('rf','gmm')

## display table of results
df_cp = pd.DataFrame(cp_results)
df_cp["n_clusters"] = [str(i) for i in np.arange(3,8)]
df_cp.set_index("n_clusters",inplace=True)
df_cp.head(n=10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>svm-kmeans</th>
      <th>svm-gmm</th>
      <th>rf-kmeans</th>
      <th>rf-gmm</th>
    </tr>
    <tr>
      <th>n_clusters</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.000</td>
      <td>0.562</td>
      <td>0.557</td>
      <td>0.562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000</td>
      <td>0.562</td>
      <td>0.551</td>
      <td>0.562</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.566</td>
      <td>0.554</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000</td>
      <td>0.562</td>
      <td>0.566</td>
      <td>0.562</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.444</td>
      <td>0.562</td>
      <td>0.551</td>
      <td>0.562</td>
    </tr>
  </tbody>
</table>
</div>



## QUESTION 4

Run an experiment to see if you can you improve on your workflow with
the addition of re-sampling techniques?


```python
## YOUR CODE HERE

```


```python

```
