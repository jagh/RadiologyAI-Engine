import os
import numpy as np
import pandas as pd
import seaborn as sns

from engine.utils import Utils

from matplotlib import pyplot as plt

## pca data projection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## tSNE data projection
from sklearn.manifold import TSNE
from sklearn import manifold, datasets

## tSNE data projection yellowbrick
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer



def plot_heatmap(metadata, visualization_folder, fig_name):
    """ Plot pyRadiomics features as heatmap and clustermap """
    ## Get correlations from input dataframe
    d = metadata
    corr = d.corr()

    # Set up the matplotlib figure, make it big!
    f, ax = plt.subplots(figsize=(15, 10))

    # Draw the heatmap using seaborn
    # sns.heatmap(corr, vmax=.8, square=True,  linewidths=.3, cmap="vlag")
    sns.heatmap(corr, square=True,  linewidths=.3, cmap="vlag")
    # plt.ax_heatmap.set_xticklabels([])
    plt.savefig(os.path.join(visualization_folder, str(fig_name + "_matrix_correlation.png") ))
    plt.show()

    ## Fixed parameters
    dd = d.iloc[:,1:100]

    ## Draw the clustermap using seaborn
    pp = sns.clustermap(dd.corr(), linewidths=.5, figsize=(13,13),  cmap="vlag")
    _ = plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(os.path.join(visualization_folder, str(fig_name + "_hierchical_cluster.png") ))
    plt.show()


def pca_2D_projection(metadata, visualization_folder, fig_name):
    """ Using PCA for data visualization
            + PCA using Python (scikit-learn) -> https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    """


    # # Data standarization
    # features = ['AA_10Percentile',	'AA_90Percentile',	'AA_Energy',	'AA_Entropy',
    # 	       'AA_InterquartileRange',	'AA_Kurtosis',	'AA_Maximum',	'AA_MeanAbsoluteDeviation',
    #            'AA_Mean',	'AA_Median',	'AA_Minimum',	'AA_Range',	'AA_RobustMeanAbsoluteDeviation',
    #            'AA_RootMeanSquared',	'AA_Skewness',	'AA_TotalEnergy',	'AA_Uniformity',	'AA_Variance',
    #
    #            'BB_Elongation',	'BB_Flatness',	'BB_LeastAxisLength',	'BB_MajorAxisLength',	'BB_Maximum2DDiameterColumn',
    #            'BB_Maximum2DDiameterRow',	'BB_Maximum2DDiameterSlice',	'BB_Maximum3DDiameter',	'BB_MeshVolume',
    #            'BB_MinorAxisLength',	'BB_Sphericity',	'BB_SurfaceArea',	'BB_SurfaceVolumeRatio',	'BB_VoxelVolume',
    #
    #            'CC_Autocorrelation',	'CC_ClusterProminence',	'CC_ClusterShade',	'CC_ClusterTendency',	'CC_Contrast',
    #            'CC_Correlation',	'CC_DifferenceAverage',	'CC_DifferenceEntropy',	'CC_DifferenceVariance',	'CC_Id',	'CC_Idm',
    #            'CC_Idmn',	'CC_Idn',	'CC_Imc1',	'CC_Imc2',	'CC_InverseVariance',	'CC_JointAverage',	'CC_JointEnergy',
    #            'CC_JointEntropy',	'CC_MCC',	'CC_MaximumProbability',	'CC_SumAverage',	'CC_SumEntropy',	'CC_SumSquares',
    #
    #            'DD_GrayLevelNonUniformity',	'DD_GrayLevelNonUniformityNormalized',	'DD_GrayLevelVariance',	'DD_HighGrayLevelRunEmphasis',
    #            'DD_LongRunEmphasis',	'DD_LongRunHighGrayLevelEmphasis',	'DD_LongRunLowGrayLevelEmphasis',	'DD_LowGrayLevelRunEmphasis',
    #            'DD_RunEntropy',	'DD_RunLengthNonUniformity',	'DD_RunLengthNonUniformityNormalized',	'DD_RunPercentage',	'DD_RunVariance',
    #            'DD_ShortRunEmphasis',	'DD_ShortRunHighGrayLevelEmphasis',	'DD_ShortRunLowGrayLevelEmphasis',
    #
    #            'EE_Busyness',	'EE_Coarseness', 'EE_Complexity',	'EE_Contrast',	'EE_Strength',
    #            'FF_DependenceEntropy',	'FF_DependenceNonUniformity',	'FF_DependenceNonUniformityNormalized',	'FF_DependenceVariance',
    #            'FF_GrayLevelNonUniformity',	'FF_GrayLevelVariance',	'FF_HighGrayLevelEmphasis',	'FF_LargeDependenceEmphasis',
    #            'FF_LargeDependenceHighGrayLevelEmphasis',	'FF_LargeDependenceLowGrayLevelEmphasis',	'FF_LowGrayLevelEmphasis',
    #            'FF_SmallDependenceEmphasis',	'FF_SmallDependenceHighGrayLevelEmphasis',	'FF_SmallDependenceLowGrayLevelEmphasis'
    #            ]

    ## Separating out the features
    # x = metadata.loc[:, features].values
    x = metadata.iloc[:,2:-1]

    ## Separating out the target
    y = metadata.loc[:,['label_name']].values

    ## Standardizing the features
    x = StandardScaler().fit_transform(x)


    ## PCA Projection to 2D
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents,
                                columns = ['principal component 1',
                                            'principal component 2'])

    finalDf = pd.concat([principalDf, metadata[['label_name']]], axis = 1)

    ## Visualize 2D Projection
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['yes', 'no']
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['label_name'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    ax.legend(targets)
    ax.grid()

    plt.savefig(os.path.join(visualization_folder, str(fig_name + "_pca_2d_projection.png") ))
    plt.show()


# from sklearn.metrics import pairwise_distances
# from sklearn.manifold import _t_sne
#
# def tSNE_2D_projection(metadata, visualization_folder, fig_name):
#     """
#     https://danielmuellerkomorowska.com/2021/01/05/introduction-to-t-sne-in-python-with-scikit-learn/
#     """
#
#     ## Separating out the features
#     X = metadata.iloc[:,2:-1].values
#
#     ## Separating out the target
#     y = metadata['label_name'].values
#
#
#
#     #########################################################################3
#     y_sorted_idc = y.argsort()
#     X_sorted = X[y_sorted_idc]
#
#     distance_matrix = pairwise_distances(X, metric='euclidean')
#     print('distance_matrix', distance_matrix)
#
#     distance_matrix_sorted = pairwise_distances(X_sorted,
#                                             metric='euclidean')
#
#     fig, ax = plt.subplots(1,2)
#     ax[0].imshow(distance_matrix, 'Greys')
#     ax[1].imshow(distance_matrix_sorted, 'Greys')
#     ax[0].set_title("Unsorted")
#     ax[1].set_title("Sorted by Label")
#
#
#     perplexity = 30  # Same as the default perplexity
#     p = _t_sne._joint_probabilities(distances=distance_matrix,
#                         desired_perplexity = perplexity,
#                         verbose=False)
#
#
#     ### Optimize Embedding with Gradient Descent
#
#     # Create the initial embedding
#     n_samples = X.shape[0]
#     n_components = 2
#     X_embedded = 1e-4 * np.random.randn(n_samples,
#                                     n_components).astype(np.float32)
#
#     embedding_init = X_embedded.ravel()  # Flatten the two dimensional array to 1D
#     print('embedding_init:', embedding_init)
#
#     # kl_kwargs defines the arguments that are passed down to _kl_divergence
#     kl_kwargs = {'P': p,
#              'degrees_of_freedom': 1,
#              'n_samples': 40,
#              'n_components':2}
#
#     # Perform gradient descent
#     embedding_done = _t_sne._gradient_descent(_t_sne._kl_divergence,
#                                           embedding_init,
#                                           0,
#                                           4,embedding_init
#                                           kwargs=kl_kwargs)
#
#     # Get first and second TSNE components into a 2D array
#     tsne_result = embedding_done[0].reshape(40, 2)
#
#     # Convert do DataFrame and plot
#     tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0],
#                                'tsne_2': tsne_result[:,1],
#                                'label': y})
#
#     fig, ax = plt.subplots(1)
#     sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
#     lim = (tsne_result.min()-5, tsne_result.max()+5)
#     ax.set_xlim(lim)
#     ax.set_ylim(lim)
#     ax.set_aspect('equal')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#
#     plt.show()


def tSNE_2D_projection_Yellowbrick(metadata, visualization_folder, fig_name):
    """ t-SNE Corpus Visualization with Yellowbrick """

    ## Separating out the features
    data = metadata.iloc[:,2:-1]

    ## Separating out the target
    # target = metadata.loc[:,['label_name']].values
    target = metadata['label_name']

    ## create document vectors
    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(data)
    y = target

    # Create the visualizer and draw the vectors
    tsne = TSNEVisualizer()
    tsne.fit(X, y)
    tsne.show()


def tSNE_2D_projection(metadata, visualization_folder, fig_name):
    """ t-SNE Corpus Visualization with Scikit Learn
            + Example: https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
    """

    ## Separating out the features
    X = metadata.iloc[:,2:-1]

    ## Separating out the target
    # target = metadata.loc[:,['label_name']].values
    y= metadata['label_name']


    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    # tsne = TSNE(n_components)
    tsne = manifold.TSNE(n_components=n_components, init='pca',
                        learning_rate=50,
                         random_state=1, perplexity=100)
    tsne_result = tsne.fit_transform(X)
    tsne_result.shape
    # (1000, 2)
    # Two dimensions for each of our images

    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1)
    # plt.figsize = (12, 12)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


# #########################################
#     sns.set(style = "ticks")
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection = '3d')
#
#     t1 = tsne_result[:,0]
#     t2 = tsne_result[:,1]
#     t3 = tsne_result[:,2]
#
#     ax.set_xlabel("Happiness")
#     ax.set_ylabel("Economy")
#     ax.set_zlabel("Health")
#
#     ax.scatter(t1, t2, t3, c=y.values, s=64)


    plt.savefig(os.path.join(visualization_folder, str(fig_name + "_tsne_2d_projection.png") ))
    plt.show()







#######################################################################
## Feature Visualization with clusters
## https://github.com/AIM-Harvard/pyradiomics/blob/master/notebooks/FeatureVisualization.ipynb
#######################################################################

## Setting the input folder
## First Exploration Experiments for pyRadiomics
# testbed_name = "LesionExt-Bern-full_cases-SK"
# testbed_name = "LesionExt-Bern-full_cases-HK"
# testbed_name = "LesionExt-Yale-10_cases"


## Second Exploration Experiments for pyRadiomics
testbed_name = "tSNE-pyRadiomics-Bern-20-cases-SK"


radiomics_folder = os.path.join("testbed", testbed_name, "radiomics_features")
visualization_folder = os.path.join("testbed", testbed_name, "visualization_features")
Utils().mkdir(radiomics_folder)
Utils().mkdir(visualization_folder)


## Read pyradiomics feature by segmentation
ggo_metadata = pd.read_csv(os.path.join(radiomics_folder, "lesion_features-0.csv") , sep=',')
con_metadata = pd.read_csv(os.path.join(radiomics_folder, "lesion_features-1.csv") , sep=',')
# ate_metadata = pd.read_csv(os.path.join(radiomics_folder, "lesion_features-2.csv") , sep=',')
print("+ ggo_metadata: ", ggo_metadata.shape)
print("+ con_metadata: ", con_metadata.shape)


## Combining the dataset into 1
# metadata = pd.concat([ggo_metadata, con_metadata, ate_metadata], axis=0)
# all_metadata = pd.concat([ggo_metadata, con_metadata], axis=0)
# print("+ metadata: ", all_metadata.head())
# print("+ metadata: ", all_metadata.shape)


#######################################################################
## launcher the visualizaion
#######################################################################

## Step-1: Compute the feature correlations
# plot_heatmap(ggo_metadata, visualization_folder, "GGO")
# plot_heatmap(con_metadata, visualization_folder, "CON")
#
# ## Combining the dataset into 1
# all_metadata = pd.concat([ggo_metadata, con_metadata], axis=0)
# print("+ metadata: ", all_metadata.head())
# print("+ metadata: ", all_metadata.shape)
# plot_heatmap(all_metadata, visualization_folder, "ALL")


# ## Step-2: PCA 2D projecttion
# pca_2D_projection(ggo_metadata, visualization_folder, "GGO")
# pca_2D_projection(con_metadata, visualization_folder, "CON")
# all_metadata = pd.concat([ggo_metadata.iloc[:,:], con_metadata.iloc[:,2:-1]], axis=1)
# pca_2D_projection(all_metadata, visualization_folder, "ALL")


## Step-3: tSNE 2D projecttion with Yellowbrick
# tSNE_2D_projection(ggo_metadata, visualization_folder, "GGO")
# tSNE_2D_projection(con_metadata, visualization_folder, "CON")
#
all_metadata = pd.concat([ggo_metadata.iloc[:,:], con_metadata.iloc[:,2:-1]], axis=1)
print("+ metadata: ", all_metadata.head())
print("+ metadata: ", all_metadata.shape)
tSNE_2D_projection(all_metadata, visualization_folder, "ALL")



## Step-2: tSNE 2D projecttion
# tSNE_2D_projection(ggo_metadata, con_metadata, all_metadata, visualization_folder, "GGO")
