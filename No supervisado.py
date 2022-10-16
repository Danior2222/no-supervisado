from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = datasets.load_iris()
x = df.data
x[:4]

x_train, x_test = train_test_split(x, test_size = 0.30, random_state = 0)

xs = x_train[:,0]
ys = x_train[:,1]
plt.scatter(xs,ys)

model = KMeans(n_clusters= 3)
model.fit(x_train)

predict = model.predict(x_train)
predict

plt.scatter(xs,ys, c=predict)
centroids = model.cluster_centers_ 

centroids_x =  centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x, centroids_y, marker='D', s= 100)
plt.show()

xs_ = x_test[:,0]
ys_ = x_test[:,1]
predict_test = model.predict(x_test)
plt.scatter(xs_,ys_, c=predict_test)

model.inertia_


ks = range(1, 6)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(x)
    inertias.append(model.inertia_)
plt.plot(ks,inertias, '-o')
plt.xlabel('Numero de clusters')
plt.ylabel('Inertia')
plt.show()

import pandas as pd

tipos = df.target
x = df.data
model = KMeans(n_clusters=3)
labels = model.fit_predict(x)
df = pd.DataFrame({'labels':labels, 'tipos':tipos})
cross_table = pd.crosstab(df['labels'], df['tipos'])
print(cross_table)
