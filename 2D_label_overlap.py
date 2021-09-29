""" Script to extract all axial with 2D shape (X, Y) slices from a 3D CT scan.
    Then, the pyradiomics features is performed.
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
# lesion_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Seg-6-Classes/"
nifti_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Data/"
lesion_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Seg-6-Classes/"

metadata_file_path = "/data/01_UB/2021-MedNeurIPS/111_dataframe_axial_slices-with-lesion.csv"

# nifti_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/axial_slicesTr-GT/"
# lesion_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/labels_slicesTr-GT/"
nifti_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/axial_slicesTs-GT/"
lesion_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/labels_slicesTs-GT/"

# general_lesion_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/general_lesionTr/"
# multiclass_lesion_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/multiclass-2_lesionTr/"
general_lesion_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/general_lesionTs/"
multiclass_lesion_slices_folder = "/data/01_UB/MedNeurIPS/2D-multiclassLesions-GT/multiclass-2_lesionTs/"



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
# preprocessing_flat = 'GENERAL'
# preprocessing_flat = 'MULTICLASS'
preprocessing_flat = 'MULTICLASS-2'

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
# for row in range(2):
    # print('row', metadata.iloc[row])

    ## locating the CT and Seg
    ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])
    lesion_nifti_file = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

    slice_position = metadata['slice_position'][row]-1
    slice_with_lesion = metadata['slice_with_lesion'][row]

    print('++ Patient ID: {}'.format(metadata['id_case'][row]))

    ct = nib.load(ct_nifti_file)
    ct_scan_array = ct.get_fdata()
    ct_scan_affine = ct.affine
    # print("++ ct_scan_affine:", ct_scan_affine.shape)

    lesion = nib.load(lesion_nifti_file)
    lesion_array = lesion.get_fdata()
    lesion_affine = lesion.affine
    # print("++ lesion_affine:", lesion_affine.shape)


    print('+ Slice Position: {}'.format(slice_position))
    # print('+ Slice With Lesion: {} \n {}'.format(slice_with_lesion, '--'*5))

    ############################################################################
    ## Set the general lesion segmentation and multiclass lesion segmentation
    ## https://www.programcreek.com/python/example/96388/SimpleITK.GetImageFromArray
    ct_slice = ct_scan_array[:, :, slice_position]
    lesion_slice = lesion_array[:, :, slice_position]
    # print("++ lesion_slice", type(lesion_slice))

    if preprocessing_flat == 'GENERAL':
        general_lesion_slice = np.zeros_like(lesion_slice)
        general_lesion_slice[lesion_slice == 0] = 0
        general_lesion_slice[lesion_slice == 1] = 1
        general_lesion_slice[lesion_slice == 2] = 1

        ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
        lesion_nifti = nib.Nifti1Image(general_lesion_slice, lesion_affine)

        ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
        lesion_slice_path = general_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

        # nib.save(ct_nifti, ct_slice_path)
        nib.save(lesion_nifti, lesion_slice_path)

        try:
            re = RadiomicsExtractor(1)
            lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
                                                                str(metadata['id_case'][row]) + str(slice_position), "None")
        except(Exception, ValueError) as e:
            re = RadiomicsExtractor(0)
            lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
                                                                str(metadata['id_case'][row]) + str(slice_position), "None")
            lesion_feature_extraction_list = np.zeros_like(lesion_feature_extraction_list)
        ## writing features by image
        csvw = csv.writer(general_features_file)
        if write_header == False:
            csvw.writerow(image_header_list)
            write_header = True
        csvw.writerow(lesion_feature_extraction_list)


    elif preprocessing_flat == 'MULTICLASS':
        multiclass_lesion_slice = np.zeros_like(lesion_slice)
        multiclass_lesion_slice[lesion_slice == 0] = 0
        multiclass_lesion_slice[lesion_slice == 1] = 1
        multiclass_lesion_slice[lesion_slice == 2] = 1
        multiclass_lesion_slice[lesion_slice == 3] = 1
        multiclass_lesion_slice[lesion_slice == 4] = 1
        multiclass_lesion_slice[lesion_slice == 5] = 1
        multiclass_lesion_slice[lesion_slice == 6] = 1

        ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
        lesion_nifti = nib.Nifti1Image(multiclass_lesion_slice, lesion_affine)

        ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
        lesion_slice_path = multiclass_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

        nib.save(ct_nifti, ct_slice_path)
        nib.save(lesion_nifti, lesion_slice_path)

        re = RadiomicsExtractor(1)
        lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
                                                                str(metadata['id_case'][row]) + str(slice_position), "None")

        ## writing features by image
        csvw = csv.writer(multiclass_features_file)
        if write_header == False:
            csvw.writerow(image_header_list)
            write_header = True
        csvw.writerow(lesion_feature_extraction_list)


    elif preprocessing_flat == 'MULTICLASS-2':


        lesions = np.unique(lesion_slice)
        lesion_list = []
        for l in lesions:
            # print("l: ", l)
            if l == 0:
                pass
            else:
                multiclass_lesion_slice = np.zeros_like(lesion_slice)
                # multiclass_lesion_slice[lesion_slice == 0] = 0
                multiclass_lesion_slice[lesion_slice == l] = 1

                ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
                lesion_nifti = nib.Nifti1Image(multiclass_lesion_slice, lesion_affine)

                ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
                lesion_slice_path = multiclass_lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '-' + str(int(l)) + '.nii.gz'

                nib.save(ct_nifti, ct_slice_path)
                nib.save(lesion_nifti, lesion_slice_path)

                re = RadiomicsExtractor(1)
                lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
                                                                str(metadata['id_case'][row]) + str(slice_position), "None")

                lesion_list.append(array(lesion_feature_extraction_list[2:]))


        array_lol = array(lesion_list)
        # print("array_lol: ", array_lol.shape)
        # print("array_lol: ", type(array_lol))
        row_average = average(array_lol, axis=0)
        # print("row_average: ", row_average)
        # print("row_average: ", row_average.shape)

        image_feature_list = ([])
        image_feature_list = np.append(image_feature_list, str(metadata['id_case'][row]) + str(slice_position))
        image_feature_list = np.append(image_feature_list, "None")
        image_feature_list = np.append(image_feature_list, row_average.tolist())
        # print("image_feature_list: ", image_feature_list)
        # print("image_feature_list: ", len(image_feature_list))

        ## writing features by image
        csvw = csv.writer(multiclass_features_file)
        if write_header == False:
            csvw.writerow(image_header_list)
            write_header = True
        csvw.writerow(image_feature_list)

    else:
        pass


    # else:
    #     ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
    #     lesion_nifti = nib.Nifti1Image(lesion_slice, lesion_affine)
    #
    #     ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
    #     lesion_slice_path = lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
    #
    #     nib.save(ct_nifti, ct_slice_path)
    #     nib.save(lesion_nifti, lesion_slice_path)
    #
    #     re = RadiomicsExtractor(lesion_area)
    #     lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
    #                                                             str(metadata['id_case'][row]) + str(slice_position), "None")
    #
    #     # show_slices([ct_slice, lesion_slice])
    #     # plt.suptitle("Slices sample")
    #     # plt.show()
    #
    #     ## writing features by image
    #     csvw = csv.writer(features_file)
    #     if write_header == False:
    #         csvw.writerow(image_header_list)
    #         write_header = True
    #     csvw.writerow(lesion_feature_extraction_list)




    # ###########################################
    # # ct_img = sitk.GetImageFromArray(ct_slice)
    # lesion_img = sitk.GetImageFromArray(lesion_slice, isVector=True)
    # print("++ lesion_img", lesion_img)








    # ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
    # lesion_nifti = nib.Nifti1Image(lesion_slice, lesion_affine)
    #
    # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
    # lesion_slice_path = lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

    # nib.save(ct_nifti, ct_slice_path)
    # nib.save(lesion_nifti, lesion_slice_path)
    #
    # re = RadiomicsExtractor(lesion_area)
    # lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
    #                                                         str(metadata['id_case'][row]) + str(slice_position), "None")

    # show_slices([ct_slice, lesion_slice, general_lesion_slice, multiclass_lesion_slice])
    # plt.suptitle("Slices sample")
    # plt.show()






# #########################################################################
# ###########################################################################
# ## iterate between cases
# # for row in range(metadata.shape[0]):
# for row in range(2):
#     imgPath = os.path.join(nifti_folder, metadata['ct_file_name'][row])
#     maskPath = os.path.join(lesion_folder, metadata['lesion_file_name'][row])
#
#     image = sitk.ReadImage(imgPath)
#     mask = sitk.ReadImage(maskPath)
#
#     label = sitk.GetArrayFromImage(mask)
#     print("+ label: ", label.shape)



# lsif = sitk.LabelShapeStatisticsImageFilter()
# lsif.Execute(mask)
# # print("+ lsif", lsif)
#
# print("lsif.GetBoundingBox(1)", lsif.GetBoundingBox(2))
# # boundingBox = np.array(lsif.GetBoundingBox(1))
# # print("+ boundingBox", boundingBox)


# label = sitk.GetArrayFromImage(mask)


#
#
# ## iterate between segmentation layers to perform feature extration
# for lesion_area in range(0, seg_layer_number):
#     ## Set file name to write a features vector per case
#     filename = str(radiomics_folder+"/lesion_features-"+str(lesion_area)+".csv")
#     features_file = open(filename, 'w+')
#
#     ## iterate between cases
#     for row in range(metadata.shape[0]):
#         # print('row', metadata.iloc[row])
#
#         ## locating the CT and Seg
#         ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])
#         lesion_nifti_file = os.path.join(lesion_folder, metadata['lesion_file_name'][row])
#
#         slice_position = metadata['slice_position'][row]-1
#         slice_with_lesion = metadata['slice_with_lesion'][row]
#
#         print('+ Patient ID: {}'.format(metadata['id_case'][row]))
#
#         if slice_with_lesion == 1:
#             ## get the ct arrays
#             # ct_scan_array, ct_scan_affine = Utils().read_nifti(ct_nifti_file)
#             # lesion_array, lesion_affine = Utils().read_nifti(lesion_nifti_file)
#
#             ct = nib.load(ct_nifti_file)
#             ct_scan_array = ct.get_fdata()
#             ct_scan_affine = ct.affine
#             print("++ ct_scan_affine:", ct_scan_affine.shape)
#
#             lesion = nib.load(lesion_nifti_file)
#             lesion_array = lesion.get_fdata()
#             lesion_affine = lesion.affine
#             print("++ lesion_affine:", lesion_affine.shape)
#
#             # print('+ CT shape: {} '.format(ct_scan_array.shape))
#             # print('+ Lesion shape {} \n{}'.format(lesion_array.shape, '--'*5))
#             print('+ Slice Position: {}'.format(slice_position))
#             print('+ Slice With Lesion: {} \n {}'.format(slice_with_lesion, '--'*5))
#
#             ct_slice = ct_scan_array[:, :, slice_position]
#             lesion_slice = lesion_array[:, :, slice_position]
#
#             ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
#             lesion_nifti = nib.Nifti1Image(lesion_slice, lesion_affine)
#
#             ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
#             lesion_slice_path = lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
#
#             nib.save(ct_nifti, ct_slice_path)
#             nib.save(lesion_nifti, lesion_slice_path)
#
#             re = RadiomicsExtractor(lesion_area)
#             lesion_feature_extraction_list, image_header_list = re.feature_extractor_2D(ct_slice_path, lesion_slice_path,
#                                                                     str(metadata['id_case'][row]) + str(slice_position), "None")
#
#             # show_slices([ct_slice, lesion_slice])
#             # plt.suptitle("Slices sample")
#             # plt.show()
#
#             ## writing features by image
#             csvw = csv.writer(features_file)
#             if write_header == False:
#                 csvw.writerow(image_header_list)
#                 write_header = True
#             csvw.writerow(lesion_feature_extraction_list)
#         else:
#             ## Healthy slices or no manual segmentations
#             pass
