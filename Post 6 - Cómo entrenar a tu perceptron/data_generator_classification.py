import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from datetime import datetime

X, y = make_blobs(n_samples=1000,n_features=2, centers=2,cluster_std=1.5)
dictdata =  dict(feature1=X[:,0], feature2=X[:,1], type=y)
df = pd.DataFrame(dictdata)

now = datetime.now()
filename = 'perceptron_data_%d%d%d.csv' %(now.year, now.month, now.day)
df.to_csv(filename, sep = ';', index = False, encoding = 'utf-8')

plt.title("Random data with 'make_blobs'", fontsize='small')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, alpha=0.5, edgecolor='c')
plt.show()