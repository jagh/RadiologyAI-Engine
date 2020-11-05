"""
Disease severity classification of lung tissue abnormalities using the MosMedata.
"""

import sys, os
import glob
import pandas as pd
import csv

from engine.utils import Utils
from engine.segmentations import LungSegmentations
from engine.featureextractor import RadiomicsExtractor




#######################################################################
## Launcher settings to classify a ground-glass opacities score
#######################################################################

## Dataset path definitions
studies_folder = glob.glob("/data/01_UB/CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/*")
testbed = "testbed/"
bilung_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_bi-lung/")
lobes_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_lobes/")


#######################################################################
## Stage-1: bi-lung and lung lobes CT segmentation
def auto_segmentation(studies_folder, segmentation_folder, seg_method):
    """ Formatting the folder studies for each GGO categories """
    for std_path in studies_folder:
        input_std = glob.glob(str(std_path + "/*"))
        folder_name = std_path.split(os.path.sep)[-1]
        output_std = os.path.join(segmentation_folder, folder_name)
        Utils().mkdir(output_std)

        ## Launch of automatic CT segmentation
        LungSegmentations().folder_segmentations(input_std, output_std, seg_method, 5)

# auto_segmentation(studies_folder, bilung_segmentation_folder, 'bi-lung')
# auto_segmentation(studies_folder, lobes_segmentation_folder, 'lobes')



#######################################################################
## Stage-2: Feature extraction with pyradiomics
studies_path = "/data/01_UB/CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/"
metadata_file = os.path.join(testbed, "mosmeddata/metadata-covid19_1110.csv")
metadata = pd.read_csv(metadata_file, sep=',')
print("metadata: ", metadata.shape)


## Crete new folder for feature extraction
radiomics_folder = os.path.join(testbed, "mosmeddata/radiomics_features")
Utils().mkdir(radiomics_folder)

## Set file name to write a features vector per case
filename = os.path.join(radiomics_folder, "radiomics_features.csv")
features_file = open(filename, 'w+')

for row in metadata.iterrows():
    ## Getting the GGO label
    label =  row[1]["category"]

    ## Locating the ct image file
    ct_image_name = row[1]["study_file"].split(os.path.sep)[-1]
    ct_image_path = os.path.join(studies_path, label, ct_image_name)
    ct_case_id = ct_image_name.split(".nii.gz")[0]

    ## Locating the bi-lung segmentation file
    bilung_segmentation_name = str(ct_case_id + "-bi-lung.nii.gz")
    bilung_segmentation_path = os.path.join(bilung_segmentation_folder, label, bilung_segmentation_name)

    ## Feature extraction by image
    re = RadiomicsExtractor()
    image_feature_list = re.feature_extractor(ct_image_path, bilung_segmentation_path, ct_case_id, label)

    ## writing features by image
    csvw = csv.writer(features_file)
    csvw.writerow(image_feature_list)


# print("metadata: ", metadata.shape)
