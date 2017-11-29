'''


In this assignment, the goal is to implement the k-Means or bisecting k-Means clustering algorithm
to cluster a set of data points in the input file (test.dat) where the file  is a simple CSR sparse
matrix containing the features associated with different feature ids.

On a high level, here, first the input data is read and preprocessed by taking into account the
frequency of each term. Once that is accounted for, it is converted into a csr matrix and k means
is tried on the matrix. To get the k means, there are two distance functions tried - based on
cosine similarity and based on euclidean distance.
After the k means is complete and the algorithm is converged, the output cluster ids are written to a file
'''

# Import needed modules
import sys
#sys.path.append('/Users/anaconda/lib/python2.7/site-packages')
#sys.path.append('/anaconda/lib/python2.7/site-packages')
from scipy.sparse import csr_matrix
import numpy as np
from lib2matrix import csr_l2normalize, build_matrix
from sklearn.decomposition import TruncatedSVD
from operator import itemgetter
import random
from sklearn import metrics

# Defining needed funtions

def cluster_points_euclidean(mat, mu):
    ''' This function takes two inputs: a matrix (mat) and an array of centroids (mu) where
    the mat is the matrix of all the data inputs formed from the input data and mu is
    a list of K centroids.
    This function calculates the minimum euclidean distance between every data point and
    every centroid and depending on which ever centroid is the closest i.e. min distance,
    the data point is assigned to that cluster id and saved in cluster_for_datapoints dictionary.
    In the clusters dictionary, the keys are the cluster ids and for each cluster id, the
    data points associated as saved as that key's values.
    Finally the dictionary is returned as the output'''
    cluster_for_datapoints = {}
    for u in range(len(mat)):
        # For every data point, calculate the min distance from each mu and saved that mukey (which is between 1 and K)
        selected_mu_key = min([(i, np.linalg.norm(mat[u]-m)) for i,m in enumerate(mu,1)], key=lambda t:t[1])[0]

        # If the selected mu key exists, append the data point. Otherwise, create the key.
        try:
            cluster_for_datapoints[selected_mu_key].append(u)
        except KeyError:
           cluster_for_datapoints[selected_mu_key] = [u]

    return cluster_for_datapoints


def cluster_points_cosine(mat, mu):
    ''' This function takes two inputs: a matrix (mat) and an array of centroids (mu) where
    the mat is the matrix of all the data inputs formed from the input data and mu is
    a list of K centroids.
    This function calculates the maximum cosine similarity between every data point and
    every centroid and depending on which ever centroid is the closest i.e. max similarity,
    the data point is assigned to that cluster id and saved in cluster_for_datapoints dictionary.
    In the clusters dictionary, the keys are the cluster ids and for each cluster id, the
    data points associated as saved as that key's values.
    Finally, the dictionary is returned as the output'''
    
    cluster_for_datapoints = {}
    for u in range(len(mat)):
        # For every data point, calculate the max similarity with each mu and saved that mukey (which is between 1 and K)
        selected_mu_key = max([(i, (np.dot(mat[u], (m.T)))) for i,m in enumerate(mu,1)], key=lambda t:t[1])[0]

        # If the selected mu key exists, append the data point. Otherwise, create the key.
        try:
            cluster_for_datapoints[selected_mu_key].append(u)
        except KeyError:
           cluster_for_datapoints[selected_mu_key] = [u]

    return cluster_for_datapoints


def find_new_mu(mat, cluster_of_data):
    '''This function takes two inputs: a matrix (mat) and a dictionary (cluster_of_data),
    where the mat is the matrix formed from the input data and the dictionary is of all
    the data points along with the centroid (mu) key currently assigned to data points.
    This function takes all the data points for each centroid, and then finds the mean
    for those points to save as the new centroid (mu) for that particular cluster.
    Once each new centroid is calculated, it is appended to a new list which is then
    returned as the output.'''
    new_mu_list = []
    keys = sorted(cluster_of_data.keys())
    for k in keys:
        points_in_cluster = cluster_of_data[k] # Points in cluster is a list of indexes of points which belong in a specific cluster
        # Example - points_in_cluster = [[6, 51, 58, 98, 104, 190, 288, 306...]
        # Get those points in as a numpy array

        # Find the mean of all the points in a cluster and append it to the list.
        new_mu_list.append(np.mean(itemgetter(points_in_cluster)(mat), axis = 0))
    return new_mu_list

def check_if_mu_has_converged(mu1, mu2):
    '''This function takes two arrays and if all the entries in the arrays are exactly
    the same, then it returns True. Othewise, it returns False.
    This function is used to check whether the k means algorithm has converged
    (where all the centroids are exactly the same before and after finding a new means)'''
    converged = (set([tuple(x) for x in mu1]) == set([tuple(x) for x in mu2])) # True if centroids are same, otherwise False
    return converged


def find_centroids_using_euclidean(mat, K):
    '''This function takes two inputs: matrix (mat) and an integer (K), where
    the matrix is the csr matrix formed using the input data and K is the number
    of clusters that the training data needs to be clustered into.
    This function uses the euclidean distance as the clustering metric.
    It returns a dictionary where each data point is classified into cluster
    using the cluster id (from 1 to K) as the key.
    To do this, first K random centroids are chosen from the matrix and then
    clusters points are calculated using euclidean distance'''

    # Initialize to K random centroids as mu1 and mu2. The K means function will run till these become same
    mu1 = random.sample(mat, K)
    mu2 = random.sample(mat, K)

    # Check if the two mu are same. If not, run the K means to find new centroids
    while not check_if_mu_has_converged(mu1, mu2):
        mu2 = mu1
        cluster_for_datapoints = cluster_points_euclidean(mat, mu1)
        mu1 = find_new_mu(mat, cluster_for_datapoints)
    print 'Centers and clusters calculated'
    return(mu1, cluster_for_datapoints)

def find_centroids_using_cosine(mat, K):
    '''This function takes two inputs: matrix (mat) and an integer (K), where
    the matrix is the csr matrix formed using the input data and K is the number
    of clusters that the training data needs to be clustered into.
    This function uses the cosine similarity as the clustering metric.
    It returns a dictionary where each data point is classified into cluster
    using the cluster id (from 1 to K) as the key.
    To do this, first K random centroids are chosen from the matrix and then
    clusters points are calculated using cosine similarity'''

    # Initialize to K random centroids as mu1 and mu2. The K means function will run till these become same
    mu1 = random.sample(mat, K)
    mu2 = random.sample(mat, K)

    # Check if the two mu are same. If not, run the K means to find new centroids
    while not check_if_mu_has_converged(mu1, mu2):
        mu2 = mu1
        cluster_for_datapoints = cluster_points_euclidean(mat, mu1)
        mu1 = find_new_mu(mat, cluster_for_datapoints)
    print 'Centers and clusters calculated'
    return(mu1, cluster_for_datapoints)


# End of defining functions.

# Start the file
if __name__ == '__main__':
    print 'Opening dataset'
    f = open ('/Users/Documents/hw6/train.dat', 'r')
    lines = f.readlines()
    f.close()
    docs = []

    # Pre processing the data set and converting it into csr matrix and normalizing it.
    print 'Converting the dataset to matrix'
    for u in range (len(lines)):
        doc = lines[u]
        doc = doc.split()
        x = len(doc)
        y = 0
        t ={}
        s = ''
        while y<x:
            t[int(doc[y])]=int(doc[y+1])
            y +=2

        for i in t.keys():
            k = i
            v = t[i]
            s = s+((str(k)+' ')*v)

        s = s.rstrip()
        l = s.split()

        docs.append(l)

    mat = build_matrix(docs)
    csr_l2normalize(mat)
    print 'Matrix created'

    # This csr matrix is highly dimensional. This will lead to poor clustering due to 'Curse of Dimensionality'
    # Applying feature reduction to still keep as much data variance as possible with relatively small number of features
    print 'Applying feature reduction'
    svd = TruncatedSVD(n_components=500)
    mat_t= svd.fit_transform(mat)
    print 'Percentage of variance explained by each of the selected components in total matrix is'
    print (svd.explained_variance_ratio_.sum())
    mat = mat_t

    # Matrix is created, now using k Means using cosine similarity as clustering metric.
    # Keeping K as 7
    print 'Clusterig using k means'
    new_mu, cluster_for_datapoints = find_centroids_using_cosine(mat, 7)
    # K means complete and centroids are now converged.


    # Writing to the output file.
    print 'writing to a file'
    c=[]

    for k in cluster_for_datapoints.keys():
        value = cluster_for_datapoints[k]
        for i in value:
            c.append((i,k))
    c = sorted(c, key=lambda tup: tup[0])
    f = open ('/Users/Documents/hw6/newtest1.dat', 'a+')
    
    for i in range (len(c)):
        f.write(str(c[i][1])+'\n')
    f.close()

    # Evaluating the performance of the clustering solution by using different values of K
    for x in range(3,23,2):
        print ('-------')
        print ('Value of K is '),x
        new_mu, cluster_for_datapoints = find_centroids_using_cosine(mat, x)
        c = []
        for k in cluster_for_datapoints.keys():
            value = cluster_for_datapoints[k]
            for i in value:
                c.append((i,k))
        c = sorted(c, key=lambda tup: tup[0])
        labels = []
        for i in range (len(c)):
            labels.append(c[i][1])
        # Using silhouette score with metric as cosine similarity
        score = metrics.silhouette_score(mat, labels, metric = 'cosine')
        print ('Silhouette Score this K is '),score
    
    print 'clustering complete'
    
