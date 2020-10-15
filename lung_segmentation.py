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
from engine.utils import Utils


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

# ## Convert CT scans
testbed = "testbed/"
dcm_folder = glob.glob(str(testbed + "/dataset_unibe/sources/*"))
nii_folder = str(testbed + "/dataset_unibe/train-nii/")

Utils().convert_dcm2nii(dcm_folder, nii_folder)


## CT lung lobes segmentation
input_folder = glob.glob(str(testbed + "/dataset_unibe/train-nii/Pat_IPF_1/*"))
output_folder = str(testbed + "/dataset_unibe/outputs/")

for input_path in input_folder:
    """
    CT lung lobes segmentation using UNet
    """

    ct_name = input_path.split(os.path.sep)[-1]
    ct_dcm_format = str(ct_name.split('.nii.gz')[0] + ".dcm")

    input_ct = sitk.ReadImage(input_path)
    result_out = ct_segmentation(input_ct)

    Utils().mkdir(output_folder)
    sitk.WriteImage(result_out, str(output_folder+"/"+ct_dcm_format))
    print("CT segmentation file: {}".format(str(output_folder+"/"+ct_dcm_format)))
