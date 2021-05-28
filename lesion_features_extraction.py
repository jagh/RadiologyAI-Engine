
import sys, os
import glob
import shutil
import pandas as pd
import numpy as np
import csv

from engine.utils import Utils
from engine.segmentations import LungSegmentations
from engine.featureextractor import RadiomicsExtractor



#######################################################################
## Feature extraction with pyradiomics
#######################################################################
## Dataset path definitions
testbed_name = "LesionExt-Bern-2_cases-HK"
nifti_folder = "/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/05_18_seg-HK-20210502/Bern-Nifti-Data-ii/"
lesion_folder = "/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/05_18_seg-HK-20210502/Bern-Nifti-Seg-ii/"

## Flag to write the header of each file
write_header = False
## {0: GGO, 1: CON, 2: ATE, 3: PLE}
seg_layer_number = 1

## Crete new folder for feature extraction
radiomics_folder = os.path.join("testbed", testbed_name, "radiomics_features")
Utils().mkdir(radiomics_folder)

metadata_file_path = "/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/05_18_seg-HK-20210502/index_file-feature_extraction-2-bern-HK.csv"
metadata = pd.read_csv(metadata_file_path, sep=',')
print("metadata: ", metadata)
print("metadata: ", metadata.shape)


## iterate between segmentation layers
for lesion_area in range(seg_layer_number):
# for lesion_area in range(1, seg_layer_number):
    ## Set file name to write a features vector per case
    filename = str(radiomics_folder+"/lesion_features-"+str(lesion_area)+".csv")
    features_file = open(filename, 'w+')

    ## iterate between cases
    for row in range(metadata.shape[0]):
        print('ID: {} || Lesion_area: {}'.format(metadata['id_case'][row], lesion_area))

        ## locating the CT and Seg
        ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])
        lesion_nifti_file = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

        re = RadiomicsExtractor(lesion_area)
        lesion_feature_extraction_list, image_header_list = re.feature_extractor(ct_nifti_file, lesion_nifti_file, metadata['id_case'][row], "None")

        ## writing features by image
        csvw = csv.writer(features_file)
        if write_header == False:
            csvw.writerow(image_header_list)
            write_header = True
        csvw.writerow(lesion_feature_extraction_list)

    write_header = False
