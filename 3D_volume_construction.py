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
## 2D Feature extraction with pyradiomics
#######################################################################
# ## Dataset path definitions
# nifti_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Data/"
# lesion_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Seg/"
# # metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_image_and_seg_3_cases-Bern.csv"
# metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_metrics_image_processing.csv"
# testbed_name = "2D-LesionExt-Bern-SK"   ## Experiment folder
#
# nifti_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Data/"
# lesion_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Seg/"


# ## Dataset path definitions
# nifti_folder = "/data/01_UB/CLINICCAI-2021/All-Nifti-Data/"
# lesion_folder = "/data/01_UB/CLINICCAI-2021/All-Nifti-Seg/"
# # metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_image_and_seg_3_cases-Bern.csv"
# metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D-All_index_metrics_image_processing-include.csv"
# testbed_name = "2D-All-MedicalImageProcessing"   ## Experiment folder
#
# nifti_slices_folder = "/data/01_UB/CLINICCAI-2021/All-Nifti-Slices-Data/"
# lesion_slices_folder = "/data/01_UB/CLINICCAI-2021/All-Nifti-Slices-Seg/"




## Dataset path definitions
nifti_folder = "/data/01_UB/Multiclass-LessionSegmentation/02_Nifti-Data"
lesion_folder = "/data/01_UB/Multiclass-LessionSegmentation/03_Nifti-Seg-6-Classes"
# metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_image_and_seg_3_cases-Bern.csv"
metadata_file_path = "/data/01_UB/Multiclass-LessionSegmentation/2D_clinical_progression-label_index.csv"
testbed_name = "2D-MulticlassLesionSegmentation"   ## Experiment folder

nifti_slices_folder = "/data/01_UB/Multiclass-LessionSegmentation/All-Nifti-Slices-Data/"
lesion_slices_folder = "/data/01_UB/Multiclass-LessionSegmentation/All-Nifti-Slices-Seg/"



# #######################################################################
# ## 2D Medical image processing
# #######################################################################
# ## Dataset path definitions
# nifti_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Data/"
# lesion_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Seg/"
# dnn_lesion_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Seg-DNN/"
#
# # metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_metrics_image_processing-JAGH.csv"
# metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_metrics_image_processing.csv"
# testbed_name = "2D-DiceCoefficient"   ## Experiment folder
#
# nifti_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Data/"
# lesion_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Seg/"
# lesion_dnn_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Seg/"


## Feature extraction parameters
write_header = False                    ## Flag to write the header of each file
seg_layer_number = 1                    ## {0: GGO, 1: CON, 2: ATE, 3: PLE}


## Crete new folder for feature extraction
radiomics_folder = os.path.join("testbed", testbed_name, "radiomics_features")
Utils().mkdir(radiomics_folder)

metadata = pd.read_csv(metadata_file_path, sep=',')
print("metadata: ", metadata.head())
print("metadata: ", metadata.shape)


## iterate between segmentation layers to perform feature extration
for lesion_area in range(0, seg_layer_number):
    ## Set file name to write a features vector per case
    filename = str(radiomics_folder+"/lesion_features-"+str(lesion_area)+".csv")
    features_file = open(filename, 'w+')

    ## iterate between cases
    for row in range(metadata.shape[0]):
        # print('row', metadata.iloc[row])

        ## locating the CT and Seg
        ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])
        lesion_nifti_file = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

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
