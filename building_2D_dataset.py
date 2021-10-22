
import logging
import argparse

import sys, os
import glob
import nibabel as nib

import pandas as pd
import numpy as np

from engine.utils import Utils
from engine.preprocessing import ImageProcessing


def convert_images(dataframe, cts_folder, seg_folder, sandbox='sandbox', split='train'):
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
        ct_file_name = os.path.join(cts_folder, metadata['ct_file_name'][row])
        lesion_file_name = os.path.join(seg_folder, metadata['lesion_file_name'][row])

        ## Fix the position of the slice and check for lesion
        axial_index = metadata['slice_position'][row]-1
        axial_with_lesion = metadata['slice_with_lesion'][row]

        if axial_with_lesion == 1:
            ct_nifti = ImageProcessing().extract_axial_slice_3D(ct_file_name, axial_index)
            lesion_nifti = ImageProcessing().extract_axial_slice_3D(lesion_file_name, axial_index)

            ## Set the file name for each axial slice and add the mode _0000 for SK
            ct_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '_0000.nii.gz'
            lesion_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '_0000.nii.gz'

            ## Write axial slices to a Nifti file for each axial slice
            nib.save(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))
            nib.save(lesion_nifti, os.path.join(axial_labels_folder, lesion_slice_name))

        else:
            ## Healthy axial slice or not GT
            print("+ Healthy axial slice or not GT: {} - {} ".format(axial_index, ct_file_name))


def generate_json(sandbox='sandbox'):
    """
    Generate a json file to feed the nnunet framework.
    The json file contains the following information:
    - name: name of the dataset
    - description: description of the dataset
    - tensorImageSize: 3D or 2D
    - reference: reference of the dataset
    - licence: license of the dataset
    - release: release of the dataset
    - modality: modalities of the dataset
    - labels: labels of the dataset
    - numTraining: number of training samples
    - numTest: number of test samples
    - training: list of training samples
    - test: list of test samples

    Parameters
    ----------
        sandbox (str): Path to the sandbox folder

    Returns
    -------
        None
    """

    ## Dataset conversion params
    imagesTr_dir            = os.path.join(sandbox, "imagesTr")
    imagesTs_dir            = os.path.join(sandbox, "imagesTs")
    modalities              = ["SK"]
    labels                  = {0: 'foreground', 1: 'GGO', 2: 'CON', #3: 'ATE',
                                                3: 'PLE', 4: 'BAN', 5: 'TBR'}
    dataset_name            = "Task115_COVID-19",
    license                 = "Hands on",
    dataset_description     = "Axial slice multi-class lesion segmentation for covid-19 patients",
    dataset_reference       = "Multiomics 2D slices",
    dataset_release         = '0.1'

    ## Get indetifier files
    train_identifiers = Utils().get_file_identifiers(imagesTr_dir)
    test_identifiers = Utils().get_file_identifiers(imagesTs_dir)

    ## Construction of json to feed the nnunet framework
    output_file = os.path.join(sandbox, "dataset.json")
    json_dict   = {}
    json_dict['name']               = dataset_name
    json_dict['description']        = dataset_description
    json_dict['tensorImageSize']    = "3D"
    json_dict['reference']          = dataset_reference
    json_dict['licence']            = license
    json_dict['release']            = dataset_release
    json_dict['modality']           = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels']             = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining']        = len(train_identifiers)
    json_dict['numTest']            = len(test_identifiers)
    json_dict['training']           = [{'image': "./imagesTr/%s.nii.gz" % i,
                                        "label": "./labelsTr/%s.nii.gz" % i} for i in train_identifiers]
    json_dict['test']               = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    ## Write output file in sandbox folder
    if not os.path.exists(output_file):
        os.path.join(sandbox, "dataset.json")
    Utils().save_json(json_dict, os.path.join(output_file))



def run(args):
    """
    Run the pipeline to build a 2D dataset from CT scans
    :param dataframe: Path to the dataframe containing the CT scans
    :param cts_folder: Path to the folder containing the CT scans
    :param seg_folder: Path to the folder containing the segmentations
    :param sandbox: Path to the sandbox where intermediate results are stored
    """
    convert_images(args.dataframe, args.cts_folder, args.seg_folder, args.sandbox, "train")
    convert_images(args.dataframe, args.cts_folder, args.seg_folder, args.sandbox, "test")
    generate_json(args.sandbox)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/00_Dev/sandbox/')
    parser.add_argument('-d', '--dataframe', default='/data/01_UB/00_Dev/2_dataframe_axial_slices.csv')
    parser.add_argument('-cts', '--cts_folder', default='/data/01_UB/00_Dev/01_Nifti-Data/')
    parser.add_argument('-seg', '--seg_folder', default='/data/01_UB/00_Dev/sandbox/00-relabel_folder/')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
