
""" """

import os
import pandas as pd
import numpy as np


## Step 1: Lung file paths
testbed = "/home/jagh/Documents/01_UB/RadiologyAI-Engine/testbed-ECR22"
lungTr_filename = os.path.join(testbed, "00_LUNG", "lesion_features-0-Tr.csv")
lungTr_dataframe = pd.read_csv(lungTr_filename, sep=',', header=0)

lungTs_filename = os.path.join(testbed, "00_LUNG", "lesion_features-0-Ts.csv")
lungTs_dataframe = pd.read_csv(lungTs_filename, sep=',', header=0)


# ## Step 2: Lesion paths
# lesionTr_filename = os.path.join(testbed, "03_MULTICLASS/radiomics_features-sources/multi2class_lesion_features-5-Tr.csv")
# lesionTr_dataframe = pd.read_csv(lesionTr_filename, sep=',', header=0)
#
# lesionTs_filename = os.path.join(testbed, "03_MULTICLASS/radiomics_features-sources/multi2class_lesion_features-5-Ts.csv")
# lesionTs_dataframe = pd.read_csv(lesionTs_filename, sep=',', header=0)

## Step 2: Overlap Lesion paths
lesionTr_filename = os.path.join(testbed, "02_GENERAL/radiomics_features-sources/general_lesion_features-Tr.csv")
lesionTr_dataframe = pd.read_csv(lesionTr_filename, sep=',', header=0)

lesionTs_filename = os.path.join(testbed, "02_GENERAL/radiomics_features-sources/general_lesion_features-Ts.csv")
lesionTs_dataframe = pd.read_csv(lesionTs_filename, sep=',', header=0)


print("+ lungTr_dataframe:", lungTr_dataframe.shape)
print("+ lungTs_dataframe:", lungTs_dataframe.shape)
print("+ lesionTr_dataframe:", lesionTr_dataframe.shape)
print("+ lesionTs_dataframe:", lesionTs_dataframe.shape)


###########################################################################
## Concatenating File 2 and File 3
## Merging Lung with lesion max_features
lwlTr_full = lungTr_dataframe.merge(lesionTr_dataframe, on='study_name', how='left')
lwlTs_full = lungTs_dataframe.merge(lesionTs_dataframe, on='study_name', how='left')

print("+ lwlTr_full:", lwlTr_full.shape)
print("+ lwlTs_full:", lwlTs_full.shape)

# ## Write the output files
# outputTr_filename = os.path.join(testbed, "03_MULTICLASS/multi2class_lwl-5-Tr-LungWFullFeatures.csv")
# lwlTr_full.to_csv(outputTr_filename, sep=',', index=False)
#
#
# outputTs_filename = os.path.join(testbed, "03_MULTICLASS/multi2class_lwl-5-Ts-LungWFullFeatures.csv")
# lwlTs_full.to_csv(outputTs_filename, sep=',', index=False)


## Write the output files
outputTr_filename = os.path.join(testbed, "02_GENERAL/general2class-Tr-LWFF.csv")
lwlTr_full.to_csv(outputTr_filename, sep=',', index=False)


outputTs_filename = os.path.join(testbed, "02_GENERAL/general2class-Ts-LWFF.csv")
lwlTs_full.to_csv(outputTs_filename, sep=',', index=False)
