import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

#Data reading and cleaning
derm = pd.read_csv('dermatology.data',header = None)
for column in derm:
    derm = derm[derm[column]!='?']
derm = derm.astype(int)

#Method1
class One():
    def __init__(self,data):
        self.data = data.drop(34,axis=1)
        self.target = data[34]

    def variance(self):
        variances = {}
        for column in self.data.columns:
            variances.update({column:np.var(self.data[column])})
        return variances

    def info_gain(self):
        res = dict(zip(self.data.columns,
               mutual_info_classif(self.data, self.target, discrete_features=True)
               ))
        return res

#Method 2
class Two():
    def __init__(self,data):
        self.data = data.drop(34,axis=1)
        self.target = data[34]

    def select(self,ForB):
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        sfs1 = sfs(clf,
           k_features=20,
           forward=ForB,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5,
           n_jobs=-1)
        sfs1.fit(self.data,self.target)
        feat_cols = list(sfs1.k_feature_idx_)
        return feat_cols
#Method 3
class Three():
    def __init__(self,data):
        self.data = data.drop(34,axis=1)
        self.target = data[34]

    def pca(self):
        #Normalizing data
        self.data = (self.data - self.data.mean())/self.data.std()
        pca = PCA()
        data_pca = pca.fit_transform(self.data)
        explained_variance = pca.explained_variance_ratio_
        print(data_pca)
        print('\n')
        print(explained_variance)
        print('\n')
        res = dict(zip(self.data.columns,explained_variance))
        return res

first  = One(derm)
variance_dic = first.variance()
print(variance_dic)
info_dic  = first.info_gain()
print(info_dic)

plt.xlabel('Features')
plt.plot(*zip(*sorted(variance_dic.items())),label='Variance')
plt.xlim(1,33)
plt.ylim(0,10)
plt.legend(loc = "upper left")
plt.savefig('Variance_method.png')

plt.plot(*zip(*sorted(info_dic.items())),label ='Info Gain')
plt.xlim(1,33)
plt.legend(loc = "upper left")
plt.savefig('Info_method.png')

second = Two(derm)
forward_features = second.select(True)
backward_features = second.select(False)

print(forward_features)
print(backward_features)

third = Three(derm)
exvar = third.pca()

plt.plot(*zip(*sorted(exvar.items())),label = 'PCA')
plt.xlim(1,33)
plt.ylim(0,2)
plt.legend(loc = "upper left")
plt.savefig('PCA_method.png')



