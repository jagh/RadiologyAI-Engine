
import logging
import argparse

import sys, os
import glob
import nibabel as nib

import pandas as pd

from engine.utils import Utils
from engine.preprocessing import SegProcessing


def dataset_conversion(dataframe, cts_folder, seg_folder,
                                            sandbox='sandbox', split='train'):
    """
    This function converts the dataset into a format that is compatible
    with the nnUNet framework. The function takes as input the metadata file,
    the folder containing the CTs, and the folder containing the manual segmentations.
    The function creates a folder for the images and another for the labels.
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

    ## Iterate between cases
    for row in range(metadata.shape[0]):

        ## Locating the CTs and labels
        ct_nifti_file = os.path.join(cts_folder, metadata['ct_file_name'][row])
        lesion_nifti_file = os.path.join(seg_folder, metadata['lesion_file_name'][row])

        ## Fix the position of the slice and check for lesion
        slice_position = metadata['slice_position'][row]-1
        slice_with_lesion = metadata['slice_with_lesion'][row]

        if slice_with_lesion == 1:
            ## get the ct array
            ct = nib.load(ct_nifti_file)
            ct_scan_array = ct.get_fdata()
            ct_scan_affine = ct.affine

            ## get the seg array
            lesion = nib.load(lesion_nifti_file)
            lesion_array = lesion.get_fdata()
            lesion_affine = lesion.affine

            ## Get the axial slice in array for images and labels
            ct_slice = ct_scan_array[:, :, slice_position]
            lesion_slice = lesion_array[:, :, slice_position]

            ## Axial slice transformation with shape (x, y, 1)
            ct_array_reshape = ct_slice.reshape((512, 512, 1))
            lesion_array_reshape = lesion_slice.reshape((512, 512, 1))

            ## Create a Nifit file for each axial slice
            ct_nifti = nib.Nifti1Image(ct_array_reshape, ct_scan_affine)
            lesion_nifti = nib.Nifti1Image(lesion_array_reshape, lesion_affine)

            ## Set the file name for each axial slice and add the mode _0000 for SK
            ct_slice_name = str(metadata['id_case'][row]) + '-' + str(slice_position) + '_0000.nii.gz'
            lesion_slice_name = str(metadata['id_case'][row]) + '-' + str(slice_position) + '_0000.nii.gz'

            ## Write axial slices to a Nifti file for each axial slice
            nib.save(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))
            nib.save(lesion_nifti, os.path.join(axial_labels_folder, lesion_slice_name))

        else:
            ## Healthy slices or no manual segmentations
            pass



def run(args):
    """ Pipeline to build an axial slice dataset """

    dataset_conversion(args.dataframe, args.cts_folder, args.seg_folder,
                                            args.sandbox, args.split)



def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/00_Dev/sandbox/')
    parser.add_argument('-d', '--dataframe', default='/data/01_UB/00_Dev/2_dataframe_axial_slices.csv')
    parser.add_argument('-cts', '--cts_folder', default='/data/01_UB/00_Dev/01_Nifti-Data/')
    parser.add_argument('-seg', '--seg_folder', default='/data/01_UB/00_Dev/sandbox/00-relabel_folder/')
    parser.add_argument('-split', '--split', default='train')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
