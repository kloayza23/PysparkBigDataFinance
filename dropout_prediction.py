
# coding: utf-8

# # Import Libraries

# In[31]:


# Load libraries
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# # Load dataset

# In[55]:
train_filename = "student_train.csv"
test_filename = "student_val.csv"
# train dataset
train_data = read_csv(train_filename, sep=";", encoding="ISO-8859-1")
X_train = train_data.loc[:,'rp_ratio_I':]
y_train = train_data.loc[:,['status']]
y_train_np = np.array(y_train).reshape(len(y_train),)
# test dataset
test_data = read_csv(test_filename, sep=";", encoding="ISO-8859-1")
X_test = test_data.loc[:,'rp_ratio_I':]
y_test = test_data.loc[:,['status']]
y_test_np = np.array(y_test).reshape(len(y_test),)


# # Analyze data

# ## Descriptive Statistics

# In[49]:


# shape
    print("X_train: ", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print ("y_test", y_test.shape)


# ## Data types of each attribute

# In[50]:


# types
set_option('display.max_rows', 500)
print("Attribute types for the X_train dataset")
print(X_train.dtypes)
print("Attribute types for the y_train dataset")
print(y_train.dtypes)


# ## Peek the data

# In[51]:


#head
set_option('display.width', 100)
print(X_train.head(20))
print(y_train.head(20))
#np.where(np.isnan(X_train))
#np.isnan(X_train)
# np.nan_to_num(X_train)
# np.where(np.isnan(X_train))
# X_train['FACTOR_INGRESO'].fillna(4)
# print(np.unique(X_train['FACTOR_INGRESO']))


# ## Summarize the distribution of each attribute

# In[52]:


# descriptions, change precision to 3 places
set_option('precision', 2)
print(X_train.describe())


# ## Class values

# In[53]:


# class distribution for the train dataset
print(y_train.groupby('status').size())
# class distribution for the test dataset
print(y_test.groupby('status').size())


# ## Unimodal Data Visualization

# In[54]:


# histograms
X_train.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()


# In[77]:


X_train.columns()


# In[56]:


# histograms
y_train.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()


# ## Density Plots

# In[76]:


# density
X_train.plot(kind='density', subplots=True, layout=(5,12), sharex=False, legend=False, fontsize=1)
pyplot.show()


# ## Multimodal Data Visualizations

# In[58]:


# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(X_train.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()


# # Evaluate Algorithms: Baseline

# ## Test parameters

# In[59]:


# Test options and evaluation metric
num_folds = 10
seed = 42
scoring = 'accuracy'


# ## Prepare algorithms to evaluate

# In[60]:


# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# ## Evaluate algorithms using the test parameters

# In[62]:


results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train_np, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('{}: {} ({})'.format(name, cv_results.mean(), cv_results.std()))


# From the results above, LR, LDA, KNN y SVM may be worth further study

# ## Compare algorithms

# In[63]:


#Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# # Evaluate Algorithms: Standardize data

# In[65]:


# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train_np, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))


# ## Compare Algorithms

# In[66]:


# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# # Algorithm Tuning

# ## Tuning SVM
# tune two key parameters of the SVM algorithm, the value of C (how much to relax the
# margin) and the type of kernel.

# In[68]:


# Tune scaled SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train_np)
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("{} ({}) with: {}".format(mean, stdev, param))


# # Finalize best model

# In[71]:


# prepare the model with the best accuracy C=0.1 and kernel = linear
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=0.1, kernel='linear')
model.fit(rescaledX, y_train_np)
# estimate accuracy on validation dataset
rescaledTestX = scaler.transform(X_test)
predictions = model.predict(rescaledTestX)
print(accuracy_score(y_test_np, predictions))
print(confusion_matrix(y_test_np, predictions))
print(classification_report(y_test_np, predictions))


# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot

X = np.array([[10, 2 , 9], [1, 4 , 3], [1, 0 , 3],
                   [4, 2 , 1], [4, 4 , 7], [4, 0 , 5], [4, 6 , 3],[4, 1 , 7],[5, 2 , 3],[6, 3 , 3],[7, 4 , 13]])
k = 3
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(k):
    # select only data observations with cluster label == i
    ds = X[np.where(labels==i)]
    print(ds.shape)
    print(ds)
    #plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()

result = zip(X , kmeans.labels_)
sortedR = sorted(result, key=lambda x: x[1])
sortedR

