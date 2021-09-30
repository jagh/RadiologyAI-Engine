""" Script to extract all axial slices with shape (X, Y) from a 3D CT scan.
    Then, is used the overlap-lesion segmentation or
    the Cov-2-Radiomics framework for feature extraction.
"""

import sys, os
import glob
import shutil
import pandas as pd
import numpy as np
import csv

from engine.utils import Utils
from engine.segmentations import LungSegmentations
from engine.featureextractor import RadiomicsExtractor
from engine.medical_image_metrics import MIM

import nibabel as nib
import SimpleITK as sitk

import matplotlib.pyplot as plt

## Medical image processing
from medpy.io import load



def read_nifti(filepath):
    '''' Reads .nii file and returns pixel array '''
    ct_scan = nib.load(filepath)
    img_array   = ct_scan.get_fdata()
    img_affine  = ct_scan.affine
    # array   = np.rot90(np.array(array))
    return (img_array, img_affine)

def show_slices(slices):
   """ Function to display row of image slices
        + Nibabel example -> https://nipy.org/nibabel/coordinate_systems.html
   """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")



#######################################################################
## Dataset path definitions
#######################################################################

#######################################################################
## 3D Folder path for Lesion Seg. and CT scans
# nifti_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Data/"
# lesion_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Seg-6-Classes/"
# # nifti_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Data/"
# # lesion_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Seg-6-Classes/"
#
# ## Experiment folder Lesion Seg. and CT scans
# testbed_name = "2D-MulticlassLesionSegmentation-GT"

#######################################################################
## 3D Folder path for Lung Seg. and CT scans
# nifti_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Data/"
# lesion_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-LungSeg/"
nifti_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Data/"
lesion_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-LungSeg/"

## Experiment folder Lung Seg. and CT scans
testbed_name = "2D-LungSegmentation"

## Output folder to host the axial slices
nifti_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/lung_axial_slicesTs/"
lesion_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/lung_labels_slicesTs/"


## Metadata with DNN and Lung index
metadata_file_path = "/data/01_UB/2021-MedNeurIPS/111_dataframe_axial_slices_with_DNN_and_Lung_index.csv"

#####################################################################
## Read the metadata
metadata_full = pd.read_csv(metadata_file_path, sep=',')
print("++++++++++++++++++++++++++++++++++++++")
# print("metadata: ", metadata_full.head())
print("++ metadata: ", metadata_full.shape)

## Using separate folder for training and test
# metadata = metadata_full.query('split == "train"')
metadata = metadata_full.query('split == "test"')
metadata = metadata.reset_index(drop=True)
# print("++ metadata:", metadata.head())
print("++ Metadata Selection:", metadata.shape)
print("++++++++++++++++++++++++++++++++++++++")


#####################################################################################
## Feature extraction parameters
write_header = False                    ## Flag to write the header of each file
seg_layer_number = 1                    ## {0: GGO, 1: CON, 2: ATE, 3: PLE}

## Crete new folder for feature extraction
radiomics_folder = os.path.join("testbed", testbed_name, "radiomics_features")
Utils().mkdir(radiomics_folder)


## iterate between segmentation layers to perform feature extration
for lesion_area in range(0, seg_layer_number):
    ## Set file name to write a features vector per case
    filename = str(radiomics_folder+"/lesion_features-"+str(lesion_area)+".csv")
    features_file = open(filename, 'w+')

    ## iterate between cases
    for row in range(metadata.shape[0]):
        # print('row', metadata.iloc[row])

        ## locating the CT
        ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])

        # ## locating the Lesion Seg.
        # lesion_nifti_file = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

        ## locating the Lung Seg.
        lesion_nifti_file = os.path.join(lesion_folder, metadata['dnn_lung_file_name'][row])

        slice_position = metadata['slice_position'][row]-1
        slice_with_lesion = metadata['slice_with_lesion'][row]

        print('+ Patient ID: {}'.format(metadata['id_case'][row]))

        if slice_with_lesion == 1:
            ## get the ct arrays
            # ct_scan_array, ct_scan_affine = Utils().read_nifti(ct_nifti_file)
            # lesion_array, lesion_affine = Utils().read_nifti(lesion_nifti_file)

            ct = nib.load(ct_nifti_file)
            ct_scan_array = ct.get_fdata()
            ct_scan_affine = ct.affine
            print("++ ct_scan_affine:", ct_scan_affine.shape)

            lesion = nib.load(lesion_nifti_file)
            lesion_array = lesion.get_fdata()
            lesion_affine = lesion.affine
            print("++ lesion_affine:", lesion_affine.shape)

            # print('+ CT shape: {} '.format(ct_scan_array.shape))
            # print('+ Lesion shape {} \n{}'.format(lesion_array.shape, '--'*5))
            print('+ Slice Position: {}'.format(slice_position))
            print('+ Slice With Lesion: {} \n {}'.format(slice_with_lesion, '--'*5))

            ct_slice = ct_scan_array[:, :, slice_position]
            lesion_slice = lesion_array[:, :, slice_position]

            ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(lesion_slice, lesion_affine)

            ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
            lesion_slice_path = lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

            nib.save(ct_nifti, ct_slice_path)
            nib.save(lesion_nifti, lesion_slice_path)

            re = RadiomicsExtractor(lesion_area)
            lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
                                                                    str(metadata['id_case'][row]) + str(slice_position), "None")

            # show_slices([ct_slice, lesion_slice])
            # plt.suptitle("Slices sample")
            # plt.show()

            ## writing features by image
            csvw = csv.writer(features_file)
            if write_header == False:
                csvw.writerow(image_header_list)
                write_header = True
            csvw.writerow(lesion_feature_extraction_list)
        else:
            ## Healthy slices or no manual segmentations
            pass
