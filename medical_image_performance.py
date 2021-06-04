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
## 2D Medical image processing
#######################################################################
## Dataset path definitions
nifti_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Data/"
lesion_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Seg/"
dnn_lesion_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Seg-DNN/"

# metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_metrics_image_processing-JAGH.csv"
metadata_file_path = "/data/01_UB/CLINICCAI-2021/2D_index_metrics_image_processing.csv"
testbed_name = "2D-DiceCoefficient"   ## Experiment folder

nifti_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Data/"
lesion_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Seg/"
lesion_dnn_slices_folder = "/data/01_UB/CLINICCAI-2021/Bern-Nifti-Slices-Seg/"

## Crete new folder for feature extraction
mim_folder = os.path.join("testbed", testbed_name, "medical_image_metrics")
Utils().mkdir(mim_folder)

output_mim_filename = str(mim_folder+"/medical_image_metrics.csv")

## Feature extraction parameters
write_header = False                    ## Flag to write the header of each file


metadata = pd.read_csv(metadata_file_path, sep=',')
print("metadata: ", metadata.head())
print("metadata: ", metadata.shape)

##
mim_file = open(output_mim_filename, 'w+')


## iterate between cases
for row in range(20):
# for row in range(metadata.shape[0]):

    ## locating the CT and Seg
    ct_nifti_file = os.path.join(nifti_folder, metadata['ct_file_name'][row])
    lesion_nifti_file = os.path.join(lesion_folder, metadata['lesion_file_name'][row])
    dnn_lesion_nifti_file = os.path.join(dnn_lesion_folder, metadata['dnn_lesion_file_name'][row])

    slice_position = metadata['slice_position'][row]
    slice_with_lesion = metadata['slice_with_lesion'][row]
    slice_with_dnn_lesion = metadata['dnn_lesion_file_name'][row]

    # print('+ Patient ID: {}'.format(metadata['id_case'][row]))
    if slice_with_lesion == 1:
        ###################
        ### https://www.programcreek.com/python/example/98177/nibabel.Nifti1Image
        ## get the ct arrays
        # ct_scan_array, ct_scan_affine = Utils().read_nifti(ct_nifti_file)
        # lesion_array, lesion_affine = Utils().read_nifti(lesion_nifti_file)

        ## Read and get ct scan array
        ct = nib.load(ct_nifti_file)
        ct_scan_array = ct.get_fdata()
        ct_scan_affine = ct.affine
        # print("++ ct_scan_affine:", ct_scan_affine.shape)

        ## Read and get lesion array
        lesion = nib.load(lesion_nifti_file)
        lesion_array = lesion.get_fdata()
        lesion_affine = lesion.affine
        # print("++ lesion_affine:", lesion_affine.shape)


        ## Read and get dnn lesion array
        dnn_lesion = nib.load(dnn_lesion_nifti_file)
        dnn_lesion_array = dnn_lesion.get_fdata()
        dnn_lesion_affine = dnn_lesion.affine
        # print("++ lesion_affine:", dnn_lesion_affine.shape)

        # print('+ CT shape: {} '.format(ct_scan_array.shape))
        # print('+ Lesion shape {} \n{}'.format(lesion_array.shape, '--'*5))
        # print('+ Slice Position: {}'.format(slice_position))
        # # print('+ Slice With Lesion: {}'.format(slice_with_lesion))
        # print('+ Slice With DNN Lesion: {} \n {}'.format(slice_with_dnn_lesion, '--'*5))

        ct_slice = ct_scan_array[:, :, slice_position-1]
        lesion_slice = lesion_array[:, :, slice_position-1]
        dnn_lesion_slice = dnn_lesion_array[:, :, slice_position-1]

        # ct_nifti = nib.Nifti1Image(ct_slice, ct_scan_affine)
        # lesion_nifti = nib.Nifti1Image(lesion_slice, lesion_affine)
        # dnn_lesion_nifti = nib.Nifti1Image(dnn_lesion_slice, dnn_lesion_affine)
        #
        # ct_slice_path = nifti_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
        # lesion_slice_path = lesion_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'
        # dnn_lesion_slice_path = lesion_dnn_slices_folder + '/' + str(metadata['id_case'][row]) + '-' + str(slice_position) + '.nii.gz'

        # nib.save(ct_nifti, ct_slice_path)
        # nib.save(lesion_nifti, lesion_slice_path)


        ####################################################################
        ## Medical image processing metric
        mim_values_list, mim_header_list = MIM().binary_metrics(dnn_lesion_slice, lesion_slice, metadata['id_case'][row], slice_position)

        # show_slices([ct_slice, lesion_slice, dnn_lesion_slice])
        # plt.suptitle("Slices sample")
        # plt.show()

        ## writing features by image
        csvw = csv.writer(mim_file)
        if write_header == False:
            csvw.writerow(mim_header_list)
            write_header = True
        csvw.writerow(mim_values_list)
    else:
        pass
