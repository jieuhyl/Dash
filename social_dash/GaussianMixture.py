# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:58:46 2018

@author: Jie.Hu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()



# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

# Plot the data with K Means Labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
                                
                                
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)                                

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

# GMM
from sklearn.mixture import GaussianMixture
gmm =  GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
        
        
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)

gmm = GaussianMixture(n_components=4, random_state=42)
plot_gmm(gmm, X)


#=============
from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1])

gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, Xmoon, label=False)


Xnew = gmm16.sample(1000)
plt.scatter(Xnew[0][:, 0], Xnew[0][:, 1])

# how many components
# covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='spherical', random_state=0).fit(Xmoon)
          for n in n_components]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.xticks(np.arange(1, 21))


# Try =========================================================================
# read data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

df_train = pd.read_csv('rsmv.csv')

df_test = pd.read_csv('it.csv')

X = df_train.iloc[:, 2:6].values

# Feature Scaling MinMaxScaler()
mm = MinMaxScaler()
X = mm.fit_transform(X)

n_components = np.arange(3, 6)
models = [GaussianMixture(n, covariance_type='full', random_state=1337).fit(X)
          for n in n_components]

plt.figure(figsize = (12, 8))
plt.plot(n_components, [m.aic(X) for m in models], label='BIC')
plt.plot(n_components, [m.bic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.xticks(n_components)

final_model = GaussianMixture(n_components[[m.bic(X) for m in models].index(min([m.bic(X) for m in models]))], 
                                            covariance_type='full', random_state=1337).fit(X)

# last
X_test = df_test.iloc[:,2:6].values
X_test = mm.transform(X_test)


gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=1337).fit(X)

from sklearn.metrics.pairwise import cosine_similarity
df = pd.concat([df_test, df_train], ignore_index=True)
X_total = np.concatenate((X_test, X), axis=0)
df['Class'] = gmm.predict(X_total)

df['Similarity_1'] = cosine_similarity(gmm.predict_proba(X_total)[0:1,:], 
                                        gmm.predict_proba(X_total))[0]

df['Similarity_2'] = cosine_similarity(X_total[0:1,:], 
                                       X_total)[0]

df['Similarity'] = (df['Similarity_1'] + df['Similarity_2'])/2

df.loc[:, 'pct_1'] = df['Similarity_1'].rank(pct=True)
df.loc[:, 'pct_2'] = df['Similarity_2'].rank(pct=True)
df.loc[:, 'pct'] = (df['pct_1'] + df['pct_2'])/2

plt.scatter(df['Similarity'], df['pct'])

#OBO
df[(df['Class'] == 3) & (df['Similarity_1'] >=0.99) & (df['Similarity_2'] >=0.99)]['OBO'].mean()
df[(df['Class'] == 3) & (df['Similarity'] >=0.99)]['OBO'].mean()

df[(df['pct_1'] >=0.95) & (df['pct_2'] >=0.95)]['OBO'].mean()
df[(df['pct'] >=0.95)]['OBO'].mean()

#================================================
ddgmm = BayesianGaussianMixture(n_components=5, covariance_type='full', weight_concentration_prior=100, 
                                weight_concentration_prior_type="dirichlet_distribution", max_iter=100, random_state=1337).fit(X)
pred = ddgmm.predict(X)
df_train['Class'] = pred
df_train['Class'].value_counts()


dpgmm = BayesianGaussianMixture(n_components=5, covariance_type='full', weight_concentration_prior=1,
                               weight_concentration_prior_type='dirichlet_process', max_iter=100, random_state=1337).fit(X)
pred = dpgmm.predict(X)
df_train['Class'] = pred
df_train['Class'].value_counts()


dpgmm.predict(X_test)
dpgmm.predict_proba(X_test)