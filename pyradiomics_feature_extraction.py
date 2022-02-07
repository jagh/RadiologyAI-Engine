import logging
import argparse

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
## Step-1: Extract features per segmentation layer between all cases
#######################################################################
def lesion_feature_extraction(input_folder,
                                sandbox,
                                dataframe_path,
                                testbed_name,
                                radiomics_set="aa"):
    """ Parallel pyradiomics feature extraction """

    nifti_folder = os.path.join(input_folder, "00_Nifti-Data_SK")
    lesion_folder = os.path.join(input_folder, "00_DNN_LesionSeg")

    ## Flag to write the header of each file
    write_header = False

    ## Crete new folder for feature extraction
    radiomics_folder = os.path.join(input_folder, "testbed", testbed_name, "radiomics_features")
    Utils().mkdir(radiomics_folder)

    metadata = pd.read_csv(os.path.join(input_folder, dataframe_path), sep=',')
    print("metadata: ", metadata)
    print("metadata: ", metadata.shape)

    # ## Set file name to write a features vector per case
    filename = str(radiomics_folder+"/lesion_features-"+ radiomics_set +".csv")
    features_file = open(filename, 'w+')

    ## iterate between cases
    # for row in range(metadata.shape[0]):
    for row in range(0, 5):
        print('+ Case ID: {}'.format(metadata['p_nr'][row]))
        # ## locating the CT and Seg
        ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])
        lesion_nifti_file = os.path.join(lesion_folder, str(metadata['p_nr'][row] + "_SK-SNF_bilungs.nii.gz" ))

        re = RadiomicsExtractor(1)
        if radiomics_set == "bb":

            ## Set filename per patient
            bb_filename_case = str(radiomics_folder+"/lesion_features-"+ radiomics_set + "-" + metadata['p_nr'][row] + ".csv")
            ## Extracting pyradiomics features
            df_shape = re.parallel_extractor(ct_nifti_file,
                                                lesion_nifti_file,
                                                metadata['p_nr'][row],
                                                metadata['who_severity_score'][row],
                                                radiomics_set,
                                                bb_filename_case
                                                )
            # print("+ Launcher | df_shape: ", df_shape)

        else:
            ## Extracting pyradiomics features
            feature_extraction_list, image_header_list = re.parallel_extractor(ct_nifti_file,
                                                                        lesion_nifti_file,
                                                                        metadata['p_nr'][row],
                                                                        metadata['who_severity_score'][row],
                                                                        radiomics_set)
            ## writing features by image
            csvw = csv.writer(features_file)
            if write_header == False:
                csvw.writerow(image_header_list)
                write_header = True
            csvw.writerow(feature_extraction_list)

    write_header = False




def run(args):
    """ """
    lesion_feature_extraction(args.input_folder, args.sandbox,
                    args.dataframe_path, args.testbed_name, args.radiomics_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/04_included_cases_FP/')
    parser.add_argument('-s', '--sandbox', default='sandbox-GGO')
    parser.add_argument('-d', '--dataframe_path', default='SNF_metadata_who-labels-radiomics_feature_extraction.csv')
    parser.add_argument('-t', '--testbed_name', default="3D_GGO4-class")
    parser.add_argument('-R', '--radiomics_set', default="gg")
                    ## 01 -> aa, 02 -> bb,  03 -> cc,  05 -> dd, 06 -> ee, 07 -> ff, 04 -> gg,
                    ### cc and dd corrupted

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
