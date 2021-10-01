""" Script to oerlap 3D multi-lesion segmentation """

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

## https://nipy.org/nibabel/reference/nibabel.cifti2.html#module-nibabel.cifti2.cifti2
from numpy import array, average



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
# nifti_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Data/"
# lesion_folder = "/data/01_UB/2021-MedNeurIPS/3D-labelsTs"
nifti_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Data/"
lesion_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Seg-6-Classes"
# lesion_folder = "/data/01_UB/2021-MedNeurIPS/3D-labelsTs"

metadata_file_path = "/data/01_UB/2021-MedNeurIPS/111_dataframe_3D_CTs.csv"

# nifti_slices_folder = "/data/01_UB/MedNeurIPS/3D-multiclassLesions/3D-CT-scansTs/"
nifti_slices_folder = "/data/01_UB/MedNeurIPS/MedicalImagingMetrics-GT/3D-CT-scansTs/"

GGO_general_lesion_folder = "/data/01_UB/MedNeurIPS/MedicalImagingMetrics-GT/MIM_3D-GGO_lesionTs/"
CON_general_lesion_folder = "/data/01_UB/MedNeurIPS/MedicalImagingMetrics-GT/MIM_3D-CON_lesionTs/"
ATE_general_lesion_folder = "/data/01_UB/MedNeurIPS/MedicalImagingMetrics-GT/MIM_3D-ATE_lesionTs/"
PLE_general_lesion_folder = "/data/01_UB/MedNeurIPS/MedicalImagingMetrics-GT/MIM_3D-PLE_lesionTs/"
BAN_general_lesion_folder = "/data/01_UB/MedNeurIPS/MedicalImagingMetrics-GT/MIM_3D-BAN_lesionTs/"
TBR_general_lesion_folder = "/data/01_UB/MedNeurIPS/MedicalImagingMetrics-GT/MIM_3D-TBR_lesionTs/"


multiclass_lesion_folder = "/data/01_UB/MedNeurIPS/3D-multiclassLesions/3D-multiclass_lesionTs/"



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


## Feature extraction parameters
write_header = False                    ## Flag to write the header of each file
seg_layer_number = 1                    ## {0: GGO, 1: CON, 2: ATE, 3: PLE}


#####################################################################################
## Crete new folder for feature extraction
preprocessing_flat = 'GENERAL'
# preprocessing_flat = 'MULTICLASS'
# preprocessing_flat = 'MULTICLASS-2'

testbed_name = preprocessing_flat   ## Experiment folder
radiomics_folder = os.path.join("testbed", testbed_name, "radiomics_features")
Utils().mkdir(radiomics_folder)



## Set file name to write a features vector per case
general_lesion_filename = str(radiomics_folder+"/general_lesion_features.csv")
general_features_file = open(general_lesion_filename, 'w+')

## Set file name to write a features vector per case
multiclass_lesion_filename = str(radiomics_folder+"/multiclass_lesion_features.csv")
multiclass_features_file = open(multiclass_lesion_filename, 'w+')


## iterate between cases
for row in range(metadata.shape[0]):
    # print('row', metadata.iloc[row])

    ## locating the CT scnas
    ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])

    ## locating the Lesion Seg.
    lesion_nifti_file = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

    ## locating the Lung Seg
    # lesion_nifti_file = os.path.join(lesion_folder, metadata['dnn_lesion_file_name'][row])
    print('++ Patient ID: {}'.format(metadata['id_case'][row]))

    ##---------------------------------------------------------
    ## 2D axial slice loading
    # slice_position = metadata['slice_position'][row]-1
    # slice_with_lesion = metadata['slice_with_lesion'][row]
    # print('+ Slice Position: {}'.format(slice_position))

    ## 3D CT scan load
    ct = nib.load(ct_nifti_file)
    ct_scan_array = ct.get_fdata()
    ct_scan_affine = ct.affine
    # print("++ ct_scan_affine:", ct_scan_affine.shape)

    ## 3D lesion scan load
    lesion = nib.load(lesion_nifti_file)
    lesion_array = lesion.get_fdata()
    lesion_affine = lesion.affine
    # print("++ lesion_affine:", lesion_affine.shape)

    ############################################################################
    ## Set the general lesion segmentation and multiclass lesion segmentation
    ## https://www.programcreek.com/python/example/96388/SimpleITK.GetImageFromArray

    ##---------------------------------------------------------
    ## 2D axial slice loading
    # ct_slice = ct_scan_array[:, :, slice_position]
    # lesion_slice = lesion_array[:, :, slice_position]

    ct_slice = ct_scan_array[:, :, :]
    lesion_slice = lesion_array[:, :, :]
    # print("++ lesion_slice", type(lesion_slice))

    if preprocessing_flat == 'GENERAL':
        #########################################################
        ## GGO_general_lesion_folder
        try:
            general_lesion_slice = np.zeros_like(lesion_slice)
            general_lesion_slice[lesion_slice == 0] = 0
            general_lesion_slice[lesion_slice == 1] = 1

            ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(general_lesion_slice, lesion_affine)

            # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
            # lesion_slice_path = general_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

            ct_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'
            lesion_path = GGO_general_lesion_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'

            nib.save(ct_nifti, ct_path)
            nib.save(lesion_nifti, lesion_path)

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")


        #########################################################
        ## CON_general_lesion_folder
        try:
            general_lesion_slice = np.zeros_like(lesion_slice)
            general_lesion_slice[lesion_slice == 0] = 0
            general_lesion_slice[lesion_slice == 2] = 1

            ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(general_lesion_slice, lesion_affine)

            # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
            # lesion_slice_path = general_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

            ct_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'
            lesion_path = CON_general_lesion_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'

            nib.save(ct_nifti, ct_path)
            nib.save(lesion_nifti, lesion_path)

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")

        #########################################################
        ## ATE_general_lesion_folder
        try:
            general_lesion_slice = np.zeros_like(lesion_slice)
            general_lesion_slice[lesion_slice == 0] = 0
            general_lesion_slice[lesion_slice == 3] = 1

            ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(general_lesion_slice, lesion_affine)

            # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
            # lesion_slice_path = general_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

            ct_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'
            lesion_path = ATE_general_lesion_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'

            nib.save(ct_nifti, ct_path)
            nib.save(lesion_nifti, lesion_path)

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")

        #########################################################
        ## PLE_general_lesion_folder
        try:
            general_lesion_slice = np.zeros_like(lesion_slice)
            general_lesion_slice[lesion_slice == 0] = 0
            general_lesion_slice[lesion_slice == 4] = 1

            ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(general_lesion_slice, lesion_affine)

            # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
            # lesion_slice_path = general_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

            ct_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'
            lesion_path = PLE_general_lesion_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'

            nib.save(ct_nifti, ct_path)
            nib.save(lesion_nifti, lesion_path)

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")


        #########################################################
        ## BAN_general_lesion_folder
        try:
            general_lesion_slice = np.zeros_like(lesion_slice)
            general_lesion_slice[lesion_slice == 0] = 0
            general_lesion_slice[lesion_slice == 5] = 1

            ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(general_lesion_slice, lesion_affine)

            # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
            # lesion_slice_path = general_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

            ct_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'
            lesion_path = BAN_general_lesion_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'

            nib.save(ct_nifti, ct_path)
            nib.save(lesion_nifti, lesion_path)

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")


        #########################################################
        ## TBR_general_lesion_folder
        try:
            general_lesion_slice = np.zeros_like(lesion_slice)
            general_lesion_slice[lesion_slice == 0] = 0
            general_lesion_slice[lesion_slice == 6] = 1

            ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(general_lesion_slice, lesion_affine)

            # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
            # lesion_slice_path = general_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

            ct_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'
            lesion_path = TBR_general_lesion_folder + '/' + str(metadata['id_case'][row]) + '.nii.gz'

            nib.save(ct_nifti, ct_path)
            nib.save(lesion_nifti, lesion_path)

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")


    ################################################################
    else:
        print("End process")


        #
        # ## 3D Feature extraction
        # re = RadiomicsExtractor(1)
        # lesion_feature_extraction_list, image_header_list = re.feature_extractor(ct_path, lesion_path, metadata['id_case'][row], "None")
        #
        # ## writing features by image
        # csvw = csv.writer(general_features_file)
        # if write_header == False:
        #     csvw.writerow(image_header_list)
        #     write_header = True
        # csvw.writerow(lesion_feature_extraction_list)


        ##---------------------------------------------------------
        ## 2D axial slice loading
        # try:
        #     re = RadiomicsExtractor(1)
        #     # lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
        #     #                                                     str(metadata['id_case'][row]) + str(slice_position), "None")
        #
        #     lesion_feature_extraction_list, image_header_list = re.feature_extractor(ct_nifti_file, lesion_nifti_file, metadata['id_case'][row], "None")
        # except(Exception, ValueError) as e:
        #     re = RadiomicsExtractor(0)
        #     lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
        #                                                         str(metadata['id_case'][row]) + str(slice_position), "None")
        #     lesion_feature_extraction_list = np.zeros_like(lesion_feature_extraction_list)


    # elif preprocessing_flat == 'MULTICLASS':
    #     multiclass_lesion_slice = np.zeros_like(lesion_slice)
    #     multiclass_lesion_slice[lesion_slice == 0] = 0
    #     multiclass_lesion_slice[lesion_slice == 1] = 1
    #     multiclass_lesion_slice[lesion_slice == 2] = 1
    #     multiclass_lesion_slice[lesion_slice == 3] = 1
    #     multiclass_lesion_slice[lesion_slice == 4] = 1
    #     multiclass_lesion_slice[lesion_slice == 5] = 1
    #     multiclass_lesion_slice[lesion_slice == 6] = 1
    #
    #     ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
    #     lesion_nifti = nib.Nifti1Image(multiclass_lesion_slice, lesion_affine)
    #
    #     ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
    #     lesion_slice_path = multiclass_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
    #
    #     nib.save(ct_nifti, ct_slice_path)
    #     nib.save(lesion_nifti, lesion_slice_path)
    #
    #     re = RadiomicsExtractor(1)
    #     lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
    #                                                             str(metadata['id_case'][row]) + str(slice_position), "None")
    #
    #     ## writing features by image
    #     csvw = csv.writer(multiclass_features_file)
    #     if write_header == False:
    #         csvw.writerow(image_header_list)
    #         write_header = True
    #     csvw.writerow(lesion_feature_extraction_list)
    #
    #
    # elif preprocessing_flat == 'MULTICLASS-2':
    #     lesions = np.unique(lesion_slice)
    #     lesion_list = []
    #     for l in lesions:
    #         # print("l: ", l)
    #         if l == 0:
    #             pass
    #         else:
    #             multiclass_lesion_slice = np.zeros_like(lesion_slice)
    #             # multiclass_lesion_slice[lesion_slice == 0] = 0
    #             multiclass_lesion_slice[lesion_slice == l] = 1
    #
    #             ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
    #             lesion_nifti = nib.Nifti1Image(multiclass_lesion_slice, lesion_affine)
    #
    #             ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
    #             lesion_slice_path = multiclass_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '-' + str(int(l)) + '.nii.gz'
    #
    #             nib.save(ct_nifti, ct_slice_path)
    #             nib.save(lesion_nifti, lesion_slice_path)
    #
    #             re = RadiomicsExtractor(1)
    #             lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
    #                                                             str(metadata['id_case'][row]) + str(slice_position), "None")
    #
    #             lesion_list.append(array(lesion_feature_extraction_list[2:]))
    #   ## end multilesion-2
        #
        # array_lol = array(lesion_list)
        # # print("array_lol: ", array_lol.shape)
        # # print("array_lol: ", type(array_lol))
        # row_average = average(array_lol, axis=0)
        # # print("row_average: ", row_average)
        # # print("row_average: ", row_average.shape)
        #
        # image_feature_list = ([])
        # image_feature_list = np.append(image_feature_list, str(metadata['id_case'][row]) + str(slice_position))
        # image_feature_list = np.append(image_feature_list, "None")
        # image_feature_list = np.append(image_feature_list, row_average.tolist())
        # # print("image_feature_list: ", image_feature_list)
        # # print("image_feature_list: ", len(image_feature_list))
        #
        # ## writing features by image
        # csvw = csv.writer(multiclass_features_file)
        # if write_header == False:
        #     csvw.writerow(image_header_list)
        #     write_header = True
        # csvw.writerow(image_feature_list)

    # else:
    #     pass
