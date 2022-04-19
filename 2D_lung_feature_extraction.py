"""
Class to extract the pyradiomics features from lung segmentations:

    1. Function to overlay the lung segmentation clases.
    2. Function to slicing the lung segmentation from GT index.
    3. Function to extract shape pyRadiomics features from lung segmentations.
    4. Function to normalize each Lesion-PixelSurface feature with the Lung-PixelSurface feature.
"""



import argparse
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
from engine.segmentations import LungSegmentations





def extract_slices_dataframe(cts_folder, seg_folder, dataframe, sandbox='sandbox', split='train', task='bilung'):
    """
    Function to slicing the lung segmentation from GT index.
    """

    metadata_full = pd.read_csv(dataframe, sep=',')

    ## Using separate folder for training and test
    if split == 'test':
        metadata = metadata_full.query('split == "test"')
        axial_slices_folder = os.path.join(sandbox, str("imagesTs"))
        axial_labels_folder = os.path.join(sandbox, str("labelsTs"))
    else:
        metadata = metadata_full.query('split == "train"')
        axial_slices_folder = os.path.join(sandbox, str("imagesTr"))
        axial_labels_folder = os.path.join(sandbox, str("labelsTr"))

    metadata = metadata.reset_index(drop=True)
    print("++ Metadata {}: {}".format(split, metadata.shape))

    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(axial_slices_folder)
    Utils().mkdir(axial_labels_folder)

    ## Iterate between axial slices
    for row in range(metadata.shape[0]):

        ## Locating the CTs and labels
        ct_file_path = os.path.join(cts_folder, metadata['ct_file_name'][row])


        if task == 'bilung':
            seg_file_name = str(metadata['id_case'][row] + "_SK-" + task + ".nii.gz")
            seg_file_path = os.path.join(seg_folder, seg_file_name)
        else:
            pass

        ## Fix the position of the slice and check for lesion
        axial_index = metadata['slice_position'][row]


        #########################################################################
        ct_nifti = ImageProcessing().extract_3D_slices(ct_file_path, axial_index)
        seg_nifti = ImageProcessing().extract_3D_slices(seg_file_path, axial_index)

        ## Set the file name for each axial slice and add the mode _0000 for SK
        ct_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '_0000.nii.gz'
        seg_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '.nii.gz'

        # ## nibabel -> Write axial slices to a Nifti file for each axial slice
        # nib.save(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))
        # nib.save(lesion_nifti, os.path.join(axial_labels_folder, lesion_slice_name))

        ## SimpleITK Write axial slices to a Nifti file for each axial slice
        sitk.WriteImage(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))
        sitk.WriteImage(seg_nifti, os.path.join(axial_labels_folder, seg_slice_name))




def run(args):
    """
    Run the pipeline to extract the pyradiomics features from lung segmentations
    """

    # DEV_slice_extraction(args.cts_folder, args.sandbox, args.task)
    extract_slices_dataframe(args.cts_folder, args.seg_folder, args.dataframe, args.sandbox, "train", args.task)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cts', '--cts_folder', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/01_Nifti_Data/')
    parser.add_argument('-seg', '--seg_folder', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/02_GT_Lung_Seg')
    parser.add_argument('-d', '--dataframe', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/00_dataframe_sources/56-cases_dataframe_slices-20220419.csv')
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/sandbox/')
    parser.add_argument('-t', '--task', default='bilung')
    # parser.add_argument('-t', '--task', default='snf-lesion')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
