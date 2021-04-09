
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

testbed_name = "yale_20cases_arno"
nifti_folder = "/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/02_20cases-20210309/YaleDataNifti-Arno/"
lesion_folder = "/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/02_20cases-20210309/YaleDataLesionSeg/"


## Crete new folder for feature extraction
radiomics_folder = os.path.join("testbed", testbed_name, "radiomics_features")
Utils().mkdir(radiomics_folder)


metadata_file = os.path.join("testbed/yale_20cases_arno", "lesion_feature_extraction.csv")
metadata = pd.read_csv(metadata_file, sep=',')
print("metadata: ", metadata.shape[0])


for lesion_area in range(6):
    print('lesion_area', lesion_area)

    ## Set file name to write a features vector per case
    filename = str(radiomics_folder+"/lesion_features-"+str(lesion_area)+".csv")
    features_file = open(filename, 'w+')

    for row in range(metadata.shape[0]):

        ## locating the CT and Seg
        ct_case_path = os.path.join(nifti_folder, str(metadata['id_case'][row] + '.nii.gz'))
        seg_case_path = os.path.join(lesion_folder, str(metadata['id_lesion_segmentation'][row] + '.nii'))
        print('seg_area', metadata['id_case'][row])

        re = RadiomicsExtractor(lesion_area)
        lesion_feature_extraction_list = re.feature_extractor(ct_case_path, seg_case_path, row, "None")


        ## writing features by image
        csvw = csv.writer(features_file)
        csvw.writerow(lesion_feature_extraction_list)



    #     lobes_area=str(lobes_area+1)
    #     filename = str(radiomics_folder+"/radiomics_features-"+lobes_area+".csv")
    #     features_file = open(filename, 'w+')
