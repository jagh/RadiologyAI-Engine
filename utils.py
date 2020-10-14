"""
Python script to define utility functions
"""

import glob
import collections
import os, pickle, math
import dicom2nifti


def _mkdir_(directory) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_dcm2niir(dcm_folder, output_folder):
    """
    Function to convert dicom files to nifti file
    """

    for folder_path in dcm_folder:
        ## Get folder name
        folder_name = folder_path.split(os.path.sep)[-1]
        # print(" - folder_name: {}".format(folder_name))

        ## Create the output folder by case
        _mkdir_(str(output_folder + folder_name ))
        nii_folder_path =  str(output_folder + "/" + folder_name + "/")
        # print(" - outpu_folder: {}".format(nii_folder_path))

        #######################################################################
        ## Converting dicom files to nifti
        #######################################################################
        dicom2nifti.convert_directory(str(folder_path + "/"), nii_folder_path)
                                                    #,reorient_nifti=True)

        #######################################################################
        ### Nii file renamed
        #######################################################################
        nii_file_name = glob.glob(str(nii_folder_path + "/*"))
        nii_new_name = str(nii_folder_path + "/" + folder_name + ".nii.gz" )
        os.rename(nii_file_name[0], nii_new_name)

        # print(" ++ nii_file_name: {}".format(nii_file_name[0]))
        # print(" ++ nii_new_name: {}".format(nii_new_name))



#######################################################################
## Function Launcher
#######################################################################

testbed = "testbed/"
dcm_folder = glob.glob(str(testbed + "/dataset_unibe/sources/*"))
output_folder = str(testbed + "/dataset_unibe/train-nii/")

convert_dcm2niir(dcm_folder, output_folder)
