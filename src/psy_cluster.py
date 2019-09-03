import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

def iterate_GMM(w,minc=1,maxc=10):
    gmms=[]
    bics=[] 
    for i in range(minc,maxc):
        gmm = GMM(w,i)
        gmms.append(gmm)
        bics.append(gmm.bic(w.T))
    bics = bics -np.min(bics)
    return gmms,bics

def GMM(w,num_clusters):
    gmm = GaussianMixture(n_components=num_clusters).fit(w.T)
    #plot_data(w,gmm.predict(w.T))
    return gmm

def plot_data(w,labels=None,plot_every=50):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    pca = PCA()
    pca.fit(w.T)    
    X = pca.transform(w.T)
    plt.title("Var. Explained: "+str(round(np.sum(pca.explained_variance_ratio_[0:3]),2)))
    if type(labels) == type(None):
        ax.scatter(X[::plot_every,0],X[::plot_every,1],X[::plot_every,2])
    else:
        marker = ['o','^','v','x','.']
        for c in range(0,len(np.unique(labels))):
            Y = X[labels==c,:]
            ax.scatter(Y[::plot_every,0],Y[::plot_every,1],Y[::plot_every,2],marker=marker[np.mod(c,len(marker))])

def split_data():
    return None


def plot_cluster(centers,weights,title):
    numc = np.shape(centers)[0]
    numr = np.shape(centers)[1]
    plt.figure()
    my_colors=['blue','green','purple','red','coral','pink','yellow','cyan','dodgerblue','peru','black','grey','violet']  
    for i in range(0,numc):
        for j in range(0,numr):
            if i ==0:
                plt.plot(i+j*0.01,centers[i,j],'o',color=my_colors[j],label=weights[j])
            else:
                plt.plot(i+j*0.01,centers[i,j],'o',color=my_colors[j])
    plt.legend()
    plt.gca().axhline(y=0,linestyle='--',color='k')
    plt.title(title)
    plt.ylim([-7,5])


