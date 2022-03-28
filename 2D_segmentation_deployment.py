"""
**2D_segmentation_deployment.py**

    Objective:
        - Generate the SNF multi-class lesion or lung segmentation in 2D axial slice files.
        - Then, execute the nnUNet to generate the segmentation outputs.
        - Finally, reconstruct the 3D segmentation per each CT, using the 3D volume affinity.

    Input:
        - CT folder, sandbox, and the segmentation task.

    Methods:
        - convert_images(...)
        - build_3D_images(...)

    Output:
        - 3D segmentation file in Nifti format.
"""




import logging
import argparse

import sys, os
import glob

import nibabel as nib
import SimpleITK as sitk
import skimage.io as io

import pandas as pd
import numpy as np

from engine.utils import Utils
from engine.preprocessing import ImageProcessing


def convert_mha_to_nii(cts_folder):
    """
    """
    nii_folder = "/data/03_MICCAI/data/nii"

    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(nii_folder)

    input_folder = glob.glob(cts_folder + "/*")

    ## Iterate between axial slices
    for input_path in input_folder[:]:

        ## Get folder name
        folder_name = input_path.split(os.path.sep)[-1]
        # print("folder_name: ", folder_name)
        print("+ input_path: ", input_path)

        ## Read the CT scan file
        img = sitk.ReadImage(input_path)

        ## Set the file name for each axial slice and add the mode _0000 for SK
        # ct_name = folder_name.split(".nii.gz")[0]
        ct_name = folder_name.split(".mha.gz")[0]
        ct_name_nii = str(ct_name + '.nii.gz')
        # print("ct_slice_name: ", ct_slice_name)

        ## SimpleITK Write axial slices to a Nifti file for each axial slice
        sitk.WriteImage(img, os.path.join(nii_folder, ct_name_nii))

def convert_images(cts_folder, sandbox, task):
    """
    """
    axial_slices_folder = os.path.join(sandbox, str("images_AxialSlices"))
    axial_labels_folder = os.path.join(sandbox, str("labels_AxialSlices"))
    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(axial_slices_folder)
    Utils().mkdir(axial_labels_folder)

    input_folder = glob.glob(cts_folder + "/*")

    ## Iterate between axial slices
    for input_path in input_folder:

        ## Get folder name
        folder_name = input_path.split(os.path.sep)[-1]
        # print("folder_name: ", folder_name)
        print("+ input_path: ", input_path)

        ## get the ct array for Nifti
        image = nib.load(input_path)
        image_array = image.get_fdata()
        image_affine = image.affine
        # print("+ image_array: ", image_array.shape[2])

        for axial_index in range(image_array.shape[2]):
            ct_nifti = ImageProcessing().extract_3D_slices(input_path, axial_index)

            ## Set the file name for each axial slice and add the mode _0000 for SK
            ct_name = folder_name.split(".nii.gz")[0]
            ct_slice_name = str(ct_name + '-' + str(axial_index) + '_0000.nii.gz')
            # print("ct_slice_name: ", ct_slice_name)

            ## SimpleITK Write axial slices to a Nifti file for each axial slice
            sitk.WriteImage(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))

def build_3D_images(cts_folder, sandbox, task='bi-lung'):
    """ """

    ## Create a directory for images and labels inside the Sandbox directory
    labelsTr_shape_path = os.path.join(sandbox, "3D_labels")
    Utils().mkdir(labelsTr_shape_path)

    input_folder = glob.glob(cts_folder + "/*")
    axial_labels_folder = os.path.join(sandbox, str("labels_AxialSlices"))

    # print("+ input_folder: ", input_folder)
    # print("+ axial_labels_folder: ", axial_labels_folder)


    ## Iterate between axial slices
    # for input_path in input_folder:
    # for input_path in input_folder[0:199]:  ##G1
    # for input_path in input_folder[199:499]:  ##G2
    # for input_path in input_folder[499:799]:  ##G3
    # for input_path in input_folder[799:999]:  ##G4
    for input_path in input_folder[999:1199]:  ##G5

        ## Get folder name
        ct_file_name = input_path.split(os.path.sep)[-1]
        case_name = ct_file_name.split(".nii.gz")[0]
        print("+ case_name: ", case_name)
        # print("+ input_path: ", input_path)
        # print("+ cts_folder: ", cts_folder)


        ## Reads .nii file and get the numpy image array
        ct = nib.load(input_path)
        ct_scan_array = ct.get_data()
        ct_affine = ct.affine


        #####################################################################
        ## New
        new_3D_lesion_array = np.zeros_like(ct_scan_array)
        # print("new_3D_lesion_array: ", new_3D_lesion_array.shape)

        for slice_position in range(ct_scan_array.shape[2]):

            ## Set file path of the 2D axial lesion
            axial_slices_per_case_file = str(axial_labels_folder + "/" + case_name + ".nii.gz-" + str(slice_position) + ".nii.gz")

            ## Reads .nii file and get the numpy image array
            axial_lesion = nib.load(axial_slices_per_case_file)
            axial_lesion_array = axial_lesion.get_data()
            axial_lesion_affine = axial_lesion.affine

            axial_lesion_reshape = np.reshape(axial_lesion_array, (512, 512))

            ## Adding slices
            new_3D_lesion_array[:, :, slice_position] = axial_lesion_reshape


            ## Added GT
            ##############################################################################
            ##############################################################################

        # print("new_3D_lesion_array: ", new_3D_lesion_array.shape)
        # new_3D_lesion_array_reshape = new_3D_lesion_array.reshape((512, 512, lesion_scan_array.shape[2]))
        # print("new_3D_lesion_array_reshape: ", new_3D_lesion_array_reshape.shape)

        new_3D_lesion_array_reshape = new_3D_lesion_array

        ## Generate the 3D lesions images
        new_3D_lesion_nifti = nib.Nifti1Image(new_3D_lesion_array_reshape, ct.affine)

        ## Set new name
        axial_slices_per_case_file = str(labelsTr_shape_path + "/" + case_name)
        new_3D_lesion_filename = str(axial_slices_per_case_file) + '-SNF_bilungs.nii.gz'

        ct_slice_output_path = os.path.join(labelsTr_shape_path, new_3D_lesion_filename)
        print("+ ct_slice_output_path: ", ct_slice_output_path)

        ## and write the nifti files
        nib.save(new_3D_lesion_nifti, ct_slice_output_path)


def run(args):
    """
    Run the pipeline to build a 2D dataset from CT scans
    :param dataframe: Path to the dataframe containing the CT scans
    :param cts_folder: Path to the folder containing the CT scans
    :param seg_folder: Path to the folder containing the segmentations
    :param sandbox: Path to the sandbox where intermediate results are stored
    """
    # convert_mha_to_nii(args.cts_folder)
    # convert_images(args.cts_folder, args.sandbox, args.task)
    build_3D_images(args.cts_folder, args.sandbox, args.task)


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--cts_folder', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/04_included_cases_FP/01_Nifti-Data_SK/')
    parser.add_argument('-c', '--cts_folder', default='/data/03_MICCAI/data/nii/')
    parser.add_argument('-s', '--sandbox', default='/data/03_MICCAI/sandbox-G5/')
    # parser.add_argument('-t', '--task', default='snf-bilung')
    parser.add_argument('-t', '--task', default='snf-lesion')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
