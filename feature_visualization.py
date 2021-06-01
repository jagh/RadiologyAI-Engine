import os
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

# from sklearn import manifold
# from sklearn.metrics import euclidean_distances
# from sklearn.decomposition import PCA

## pca data visualization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from engine.utils import Utils



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


    ## Data standarization
    features = ['AA_10Percentile',	'AA_90Percentile',	'AA_Energy',	'AA_Entropy',
    	       'AA_InterquartileRange',	'AA_Kurtosis',	'AA_Maximum',	'AA_MeanAbsoluteDeviation',
               'AA_Mean',	'AA_Median',	'AA_Minimum',	'AA_Range',	'AA_RobustMeanAbsoluteDeviation',
               'AA_RootMeanSquared',	'AA_Skewness',	'AA_TotalEnergy',	'AA_Uniformity',	'AA_Variance',

               'BB_Elongation',	'BB_Flatness',	'BB_LeastAxisLength',	'BB_MajorAxisLength',	'BB_Maximum2DDiameterColumn',
               'BB_Maximum2DDiameterRow',	'BB_Maximum2DDiameterSlice',	'BB_Maximum3DDiameter',	'BB_MeshVolume',
               'BB_MinorAxisLength',	'BB_Sphericity',	'BB_SurfaceArea',	'BB_SurfaceVolumeRatio',	'BB_VoxelVolume',

               'CC_Autocorrelation',	'CC_ClusterProminence',	'CC_ClusterShade',	'CC_ClusterTendency',	'CC_Contrast',
               'CC_Correlation',	'CC_DifferenceAverage',	'CC_DifferenceEntropy',	'CC_DifferenceVariance',	'CC_Id',	'CC_Idm',
               'CC_Idmn',	'CC_Idn',	'CC_Imc1',	'CC_Imc2',	'CC_InverseVariance',	'CC_JointAverage',	'CC_JointEnergy',
               'CC_JointEntropy',	'CC_MCC',	'CC_MaximumProbability',	'CC_SumAverage',	'CC_SumEntropy',	'CC_SumSquares',

               'DD_GrayLevelNonUniformity',	'DD_GrayLevelNonUniformityNormalized',	'DD_GrayLevelVariance',	'DD_HighGrayLevelRunEmphasis',
               'DD_LongRunEmphasis',	'DD_LongRunHighGrayLevelEmphasis',	'DD_LongRunLowGrayLevelEmphasis',	'DD_LowGrayLevelRunEmphasis',
               'DD_RunEntropy',	'DD_RunLengthNonUniformity',	'DD_RunLengthNonUniformityNormalized',	'DD_RunPercentage',	'DD_RunVariance',
               'DD_ShortRunEmphasis',	'DD_ShortRunHighGrayLevelEmphasis',	'DD_ShortRunLowGrayLevelEmphasis',

               'EE_Busyness',	'EE_Coarseness', 'EE_Complexity',	'EE_Contrast',	'EE_Strength',
               'FF_DependenceEntropy',	'FF_DependenceNonUniformity',	'FF_DependenceNonUniformityNormalized',	'FF_DependenceVariance',
               'FF_GrayLevelNonUniformity',	'FF_GrayLevelVariance',	'FF_HighGrayLevelEmphasis',	'FF_LargeDependenceEmphasis',
               'FF_LargeDependenceHighGrayLevelEmphasis',	'FF_LargeDependenceLowGrayLevelEmphasis',	'FF_LowGrayLevelEmphasis',
               'FF_SmallDependenceEmphasis',	'FF_SmallDependenceHighGrayLevelEmphasis',	'FF_SmallDependenceLowGrayLevelEmphasis'
               ]

    ## Separating out the features
    x = metadata.loc[:, features].values

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
    fig = plt.figure(figsize = (8,8))
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


    # ## Compute similarities with eculidean distances
    # metadata_np = metadata.values
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



    # ##############################################################
    # ## 2D Visualization
    # fig = plt.figure(1)
    # ax = plt.axes([0., 0., 1., 1.])
    #
    # s = 100
    #
    # ggo = [0, 1, 2]
    # con = [3, 4, 5]
    #
    # plt.scatter(pos[ggo, 0], pos[ggo, 1], color='navy', alpha=1.0, s=s, lw=1, label='ggo')
    # plt.scatter(pos[con, 0], pos[con, 1], color='turquoise', alpha=1.0, s=s, lw=1, label='con')
    #
    # plt.legend(scatterpoints=1, loc=5, shadow=False)
    #
    # similarities = similarities.max() / similarities * 100
    # similarities[np.isinf(similarities)] = 0
    # plt.show()




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
all_metadata = pd.concat([ggo_metadata, con_metadata], axis=0)
print("+ metadata: ", all_metadata.head())
print("+ metadata: ", all_metadata.shape)


#######################################################################
## launcher the visualizaion
#######################################################################

## Step-1: Compute the feature correlations
# plot_heatmap(ggo_metadata, visualization_folder, "GGO")
# plot_heatmap(con_metadata, visualization_folder, "CON")
# plot_heatmap(all_metadata, visualization_folder, "ALL")


## Step-2: PCA 2D projecttion
pca_2D_projection(ggo_metadata, visualization_folder, "GGO")
# pca_2D_projection(con_metadata, visualization_folder, "CON")
# pca_2D_projection(all_metadata, visualization_folder, "ALL")
