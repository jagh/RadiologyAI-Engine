"""
Pipeline launcher to extract the pyradiomics features from lung segmentations:

    1. Function to subtract the lesion segmentation from the lung segmentation.
"""



import argparse
from ctypes import util
import json
import logging

import sys, os
import glob
import shutil
import nibabel as nib
import SimpleITK as sitk

import pandas as pd
import numpy as np
import csv

from engine.utils import Utils
from engine.preprocessing import ImageProcessing
from engine.preprocessing import SegProcessing
from engine.featureextractor import RadiomicsExtractor
        

def folder_subtractSeg(sandbox, lesion_folder, lung_folder, output_folder):
    """
    """

    input_folder = glob.glob(sandbox + "/04_2nd-Nifti_Data/*")

    ip = SegProcessing()
    for input_path in input_folder:

        ## Get file name
        file_name = input_path.split(os.path.sep)[-1]
        file_name, _ = file_name.split(".nii.gz")
        # print("+ file_name:", file_name)

        ## Set lesion segmentation file path
        lesion_file_path = os.path.join(lesion_folder, file_name + "-SNF_bilungs.nii.gz")
        # print("+ lesion_file_path:", lesion_file_path)

        ## Set lung segmentation file path
        lung_file_path = os.path.join(lung_folder, file_name + ".nii.gz-bi-lung.nii.gz")
        # print("+ lung_file_path:", lung_file_path)    

        ## 3D lesion scan load
        lesion = nib.load(lesion_file_path)
        lesion_array = lesion.get_fdata()
        lesion_affine = lesion.affine

        ## relabeled lung segmentation load
        new_lung_nifti = ip.label_overlap(lung_file_path, sequenceOn = [1, 2], sequenceOff = [0])
        new_lung_nifti_array = new_lung_nifti.get_fdata()

        ## Subtract lesion segmentation from lung segmentation
        subtracted_array = ip.subtractSeg(lesion_array, new_lung_nifti_array)
        # print("+ subtracted_array:", subtracted_array.shape)

        ## Save subtracted segmentation
        output_file_path = os.path.join(output_folder, file_name + "-subtracted.nii.gz")
        nib.save(nib.Nifti1Image(subtracted_array, lesion_affine), output_file_path)


def run(args):
    """
    Run the pipeline to extract the pyradiomics features from lung segmentations
    """

    # Step-1: Overlay the lung segmentation clases.
    folder_subtractSeg(args.sandbox,
                            args.lesion_folder, 
                            args.lung_folder, 
                            args.output_folder,)


def main():
    ## Args to execute the step 1 and 2
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sandbox', default='//data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/')
    parser.add_argument('-le', '--lesion_folder', default='//data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/04_2nd-testset_Lesion_seg/')
    parser.add_argument('-lu', '--lung_folder', default='//data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/04_2nd-testset_Lung_seg/')
    ## 01-GGO, 02-CON, 03-PLE, 04-BAN, 05-General
    parser.add_argument('-o', '--output_folder', default='//data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/04_2nd-testset_Subtract-Lesion/')
    # parser.add_argument('-d', '--dataframe', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/AssessNet-19_RAI-Dataset/00_dataframe_sources/03_LH-testset_36_subjects_axial_slices.csv')
    # parser.add_argument('-s', '--sandbox', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/sandbox-B-2nd-testset/')
    # parser.add_argument('-t', '--task', default='SNF-BAN')

    args = parser.parse_args()
    run(args)



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
