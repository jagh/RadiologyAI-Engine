
import logging
import argparse

import sys, os
import glob
import nibabel as nib

import pandas as pd
import numpy as np

from engine.utils import Utils
from engine.preprocessing import ImageProcessing


def convert_axial_slices(dataframe, cts_folder, sandbox='sandbox', split='train'):
    """
    This function converts the dataset into a format that is compatible with the nnUNet framework.
    The function takes as input the metadata file, the folder containing the CTs, and
    the folder containing the manual segmentations.
    Finally, the function iterates over the metadata file and creates a Nifit file
    for each axial slice with shape (x, y, 1).

    Parameters
    ----------
        dataframe (str): The metadata file.
        cts_folder (str): The folder containing the CTs.
        seg_folder (str): The folder containing the GT.
        sandbox (str): Output folder.
        split (str): train or test.
    Returns:
    -------
        None
    """

    metadata_full = pd.read_csv(dataframe, sep=',')

    ## Using separate folder for training and test
    if split == 'test':
        metadata = metadata_full.query('split == "test"')
        axial_slices_folder = os.path.join(sandbox, str("All_axial_imagesTs"))
        # axial_labels_folder = os.path.join(sandbox, str("axial_labelsTs"))
    else:
        metadata = metadata_full.query('split == "train"')
        axial_slices_folder = os.path.join(sandbox, str("All_axial_imagesTr"))
        # axial_labels_folder = os.path.join(sandbox, str("labelsTr"))

    ## Remove axial slices rows and resert index
    metadata = metadata.drop_duplicates(subset=['id_case'])
    metadata = metadata.reset_index(drop=True)

    print("++ Metadata {}: {}".format(split, metadata.shape))

    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(axial_slices_folder)

    ## Iterate between axial slices
    for row in range(6, metadata.shape[0]):
        ## Locating the CTs and labels
        ct_file_name = os.path.join(cts_folder, metadata['ct_file_name'][row])
        print("+ ct_file_name: ", ct_file_name)

        ## get the ct array
        image = nib.load(ct_file_name)
        image_array = image.get_fdata()

        ## Loop into all axial slices by CT scan
        for axial_index in range(image_array.shape[2]):

            ct_nifti = ImageProcessing().extract_axial_slice_3D(ct_file_name, axial_index)

            ## Set the file name for each axial slice and add the mode _0000 for SK
            ct_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '_0000.nii.gz'

            ## Write axial slices to a Nifti file for each axial slice
            nib.save(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))


def run(args):
    """
    Run the pipeline to build a 2D dataset from CT scans
    :param dataframe: Path to the dataframe containing the CT scans
    :param cts_folder: Path to the folder containing the CT scans
    :param seg_folder: Path to the folder containing the segmentations
    :param sandbox: Path to the sandbox where intermediate results are stored
    """
    convert_axial_slices(args.dataframe, args.cts_folder, args.sandbox, "train")
    convert_axial_slices(args.dataframe, args.cts_folder, args.sandbox, "test")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge')
    parser.add_argument('-d', '--dataframe', default='/home/jagh/Documents/01_conferences_submitted/00_MIA-Explainable_2021/dataset/122_dataframe_axial_slices.csv')
    parser.add_argument('-cts', '--cts_folder', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/02_Step-3_122-CasesSegmented/01_Nifti-Data/')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
