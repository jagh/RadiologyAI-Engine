"""
Lung lobes segmentation using a lungmask module
"""

import sys, os
import glob
import re

import pandas  as pd
import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt

import SimpleITK as sitk
from third_party.lungmask import mask
from third_party.lungmask import resunet


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

def ct_segmentation(input_ct):
    """
    CT lung lobes segmentation using UNet
    """
    model = mask.get_model('unet','LTRCLobes')
    ct_segmentation = mask.apply(input_ct, model, batch_size=100)

    ### Write segmentation
    result_out = sitk.GetImageFromArray(ct_segmentation)
    result_out.CopyInformation(input_ct)
    # result_out = np.rot90(np.array(result_out)) ## modifyed the orientation

    return result_out




#######################################################################
## Launcher settings
#######################################################################

folder_input_path = glob.glob("testbed/dataset_unibe/train-nii/Pat_Control_1/*")
folder_output_path = "testbed/dataset_unibe/outputs/"

for input_path in folder_input_path:
    """
    CT lung lobes segmentation using UNet
    """

    ct_name = input_path.split(os.path.sep)[-1]
    ct_dcm_format = str(ct_name.split('.nii.gz')[0] + ".dcm")

    # print("ct_name: {}".format(input_path))
    # print("ct_dcm_format: {}".format(ct_dcm_format))

    input_ct = sitk.ReadImage(input_path)
    result_out = ct_segmentation(input_ct)
    sitk.WriteImage(result_out, str(folder_output_path+"/"+ct_dcm_format))
    print("CT segmentation file: {}".format(str(folder_output_path+"/"+ct_dcm_format)))
