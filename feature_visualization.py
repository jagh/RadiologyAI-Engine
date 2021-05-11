import os
import numpy as np
import pandas as pd
import seaborn as sns

from engine.utils import Utils


def download_data_example(filename):
    url = "http://www.spl.harvard.edu/publications/bitstream/download/5270"

    if not os.path.isfile(filename):
        if not os.path.isdir('example_data'):
            os.mkdir('example_data')
        print ("retrieving")
        urllib.request.urlretrieve(url, filename)
    else:
        print ("file already downloaded")

    extracted_path = 'example_data/tumorbase'
    if not os.path.isdir(extracted_path):
        print ("unzipping")
        z = zipfile.ZipFile(filename)
        z.extractall('example_data')
        print ("done unzipping")




#######################################################################
## Feature Visualization with clusters
## https://github.com/AIM-Harvard/pyradiomics/blob/master/notebooks/FeatureVisualization.ipynb
#######################################################################

## Setting the input folder
testbed_name = "LesionExt-Bern-23_cases"
radiomics_folder = os.path.join("testbed", testbed_name, "radiomics_features")
Utils().mkdir(radiomics_folder)



## Read pyradiomics feature by segmentation
ggo_metadata = pd.read_csv(os.path.join(radiomics_folder, "lesion_features-0.csv") , sep=',')
con_metadata = pd.read_csv(os.path.join(radiomics_folder, "lesion_features-1.csv") , sep=',')
# ate_metadata = pd.read_csv(os.path.join(radiomics_folder, "lesion_features-2.csv") , sep=',')

print("+ ggo_metadata: ", ggo_metadata.shape)
print("+ con_metadata: ", con_metadata.shape)
# print("+ ate_metadata: ", ate_metadata.shape)


## Combining the dataset into 1
# metadata = pd.concat([ggo_metadata, con_metadata, ate_metadata], axis=0)
metadata = pd.concat([ggo_metadata, con_metadata], axis=0)

print("+ metadata: ", metadata)
# print("+ metadata: ", metadata.columns[2:])
print("+ metadata: ", metadata.shape)



#######################################################################
## Multidimensional scaling
#######################################################################

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

metadata_np = metadata.values
# # print("++ Metadata: ", metadata_np[2, 2:])
#
# similarities = euclidean_distances(metadata_np[:, 2:])
#
# seed = np.random.RandomState(seed=3)
#
# mds = manifold.MDS(n_components=2, max_iter=5000, eps=1e-12, random_state=seed,
#                    n_init=10,
#                    dissimilarity="precomputed", n_jobs=1, metric=False)
# pos = mds.fit_transform(similarities)


#######################################################################
### Plot
#######################################################################
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# fig = plt.figure(1)
# ax = plt.axes([0., 0., 1., 1.])
#
# s = 100
#
# ggo = [0, 1, 2]
# con = [3, 4, 5]
# ate = [6, 7, 8]
#
# plt.scatter(pos[ggo, 0], pos[ggo, 1], color='navy', alpha=1.0, s=s, lw=1, label='ggo')
# plt.scatter(pos[con, 0], pos[con, 1], color='turquoise', alpha=1.0, s=s, lw=1, label='con')
# plt.scatter(pos[ate, 0], pos[ate, 1], color='darkorange', alpha=0.5, s=s, lw=1, label='ate')
#
# plt.legend(scatterpoints=1, loc=5, shadow=False)
#
# similarities = similarities.max() / similarities * 100
# similarities[np.isinf(similarities)] = 0
# plt.show()



#######################################################################
### Plot features as heatmap
#######################################################################


# Construct a pandas dataframe from the samples
d = pd.DataFrame(data=metadata_np[:, 2:], columns=metadata.columns[2:])

# d = ggo_metadata
d = con_metadata
corr = d.corr()

# Set up the matplotlib figure, make it big!
f, ax = plt.subplots(figsize=(10, 10))

# Draw the heatmap using seaborn
sns.heatmap(corr, vmax=.8, square=True,  cmap="vlag")
plt.show()


dd = d.iloc[:,1:50]

pp = sns.clustermap(dd.corr(), linewidths=.5, figsize=(13,13),  cmap="vlag")
_ = plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)
plt.show()
