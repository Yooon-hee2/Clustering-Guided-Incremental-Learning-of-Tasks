from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from math import*

class Discriminator(object):
    """discriminates which task to use."""

    def __init__(self, previous_samples, unrelated_tasks):

        self.previous_samples = previous_samples
        if unrelated_tasks is None:
            self.dataset_idx = 2  #ignore backbone parameters
        else:
            self.dataset_idx = len(unrelated_tasks) + 2
        self.pca_data = None
        if unrelated_tasks is None:
            self.unrelated_tasks = {}
        else:
            self.unrelated_tasks = unrelated_tasks

    def euclidean_distance(self, x, y):
        """function for calculate distance."""
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    def reduce_dimension(self, data_loader):
        """reduce dimension by PCA."""
        
        self.sample_num = data_loader.batch_size

        for idx, (image, label) in enumerate(data_loader):
            if idx == 0:
                samples = image
                samples = samples.reshape(-1, samples.shape[1]*samples.shape[2]*samples.shape[3])
                break
        
        if self.previous_samples is None:
            self.previous_samples = samples.cpu()
        else:
            self.previous_samples = np.concatenate((self.previous_samples, samples), axis=0)

        print(self.previous_samples.shape)

        pca = PCA(n_components=30) #reduce to 30-dimensional

        pre_pca_data = self.previous_samples
        self.pca_data = pca.fit_transform(pre_pca_data)
        
    def cluster(self):
        """cluster datasets by using k-means algorithm."""

        num_of_task = self.dataset_idx - 1

        kmeans = KMeans(n_clusters=num_of_task).fit(self.pca_data) #Do clustering except backbone dataset

        G = len(np.unique(kmeans.labels_)) #Number of labels

        #2D matrix  for an array of indexes of the given label
        cluster_index= [[] for i in range(G)]
        for i, label in enumerate(kmeans.labels_, 0):
            for n in range(G):
                if label == n:
                    cluster_index[n].append(i)
                else:
                    continue

        result = []

        for c in range(num_of_task):
            temp = []
            for task in range(num_of_task):
                count = 0
                for num in cluster_index[c]:
                    if num >= task * self.sample_num and num < (task + 1) * self.sample_num:
                        count += 1
                temp.append(count)
            result.append(temp)

        max_group_idx = 0 
        max_value = 0

        #find group that current task is majority
        for x in range(len(result)):
            if result[x][num_of_task-1] > max_value:
                max_group_idx = x
                max_value = result[x][num_of_task-1]

        threshold = 100
        unrelated_tasks = []

        for idx, center in enumerate(kmeans.cluster_centers_):
            standard = kmeans.cluster_centers_[max_group_idx]
            print(self.euclidean_distance(standard, center))

            if self.euclidean_distance(standard, center) > threshold:
                
                candidate = result[idx].index(max(result[idx])) + 2
                if candidate != self.dataset_idx:
                    unrelated_tasks.append(candidate)

            unrelated_tasks = list(set(unrelated_tasks))
        self.unrelated_tasks[self.dataset_idx] = unrelated_tasks

        # print(self.unrelated_tasks)
        return unrelated_tasks