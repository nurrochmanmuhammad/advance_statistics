import numpy as np
 
n = int(input())
ls0 = []
ls1 = []
matrix = []
for i in range(3):
    matrix.append([float(i) for i in input().split()])
matrix=np.array(matrix)#points need to be clustered
 
X = np.array([[0, 0], [2, 2],[0,2]], dtype='float64') #initiate centroid
 
'''
input
3
1 0
0 .5
4 0
'''
 
 
def calc_distance(X1, X2):
    return(sum((X1 - X2)**2))**0.5
 
def findClosestCentroids(ic, X):
    assigned_centroid = []
    for i in X:
        distance=[]
        for j in ic:
            distance.append(calc_distance(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid
 
def calc_mean(get_centroids,matrix):
  new_centroid=[]
  for i in np.unique(get_centroids):
    idx=[x for x, y in enumerate(get_centroids) if y ==i]
    len_idx=len(idx)
    if len_idx>1:
      sum_coord=[]
      for j in idx:
        sum_coord.append(matrix[j])
      centr=np.sum(np.array(sum_coord),axis=1)/len_idx
      new_centroid.append(centr)
      print(centr)
    else: 
      centr=matrix[idx[0]]
      new_centroid.append(centr)
      print(centr)
  return np.array(new_centroid)
#after input we will execute kmeans to find new centroids
calc_mean(get_centroids,matrix)
