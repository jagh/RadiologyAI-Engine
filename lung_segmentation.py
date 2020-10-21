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

from radiomics import featureextractor
import six

import csv




def lung_lobes_segmentation(input_ct):
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


def loop_segmentation(input_folder, output_folder):
    """
    Multiple CT lung lobes segmentation using UNet from an input folder
    """

    for input_path in input_folder:

        ct_name = input_path.split(os.path.sep)[-1]
        ct_dcm_format = str(ct_name.split('.nii.gz')[0] + "-lung_lobes.nii.gz")


        input_ct = sitk.ReadImage(input_path)
        result_out = lung_lobes_segmentation(input_ct)

        Utils().mkdir(output_folder)
        sitk.WriteImage(result_out, str(output_folder+"/"+ct_dcm_format))
        print("CT segmentation file: {}".format(str(output_folder+"/"+ct_dcm_format)))





#######################################################################
## Workflow Launcher settings
#######################################################################

#######################################################################
## Convert CT scans
testbed = "testbed/"
dcm_folder = glob.glob(str(testbed + "/dataset_unibe/sources/*"))
nii_folder = str(testbed + "/dataset_unibe/train-nii/")

Utils().convert_dcm2nii(dcm_folder, nii_folder)


#######################################################################
## CT lung lobes segmentation
input_folder = glob.glob(str(testbed + "/dataset_unibe/train-nii/Pat_IPF_1/*"))
output_folder = str(testbed + "/dataset_unibe/outputs/")

loop_segmentation(input_folder, output_folder)


#######################################################################
## Feature extraction with pyradiomics
ct_image_path = str(testbed + "/dataset_unibe/train-nii/Pat_IPF_1/9_thorax_exsp_lf__10__i70f__3_lcad.nii.gz")
mask_path = str(testbed + "/dataset_unibe/outputs/9_thorax_exsp_lf__10__i70f__3_lcad-lung_lobes.nii.gz")

params = os.path.join("engine", "pyradiomics_params.yaml")
extractor = featureextractor.RadiomicsFeatureExtractor(params)


## Calculate the feature (Segment-based)
feature_values = []
result = extractor.execute(ct_image_path, mask_path)
for key, val in six.iteritems(result):
  # feature_values[str(key)].append(val)
  print("\t%s: %s" %(key, val))
  feature_values.append(val)


## Writing the pyradiomics features
radiomics_folder = str(testbed + "/dataset_unibe/radiomics_features/")
filename = os.path.join(radiomics_folder, "9_thorax_exsp_lf__10__i70f__3_lcad-radiomics_features.csv")

with open(filename, 'w+') as f:
    csvw = csv.writer(f)
    csvw.writerow(feature_values)
