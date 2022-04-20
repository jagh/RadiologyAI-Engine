"""
Pipeline launcher to extract the pyradiomics features from lung segmentations:

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
from engine.preprocessing import SegProcessing
from engine.featureextractor import RadiomicsExtractor

from radiomics import base, cShape, deprecated


def folder_3D_label_overlap(seg_folder, sandbox, label_name="SNF_lung", output_folder="00_GT_Lung_Seg"):
    """
    This function takes in a folder of nifti files and relabels the lesion segmentation with a new label name,
    changing the label values according to the sequence.

        :param input_folder: A folder of nifti files
        :type input_folder: str

        :param label_name: The new label name
        :type label_name: str

        :return: A folder of relabeled nifti files
        :rtype: str
    """

    input_folder = glob.glob(seg_folder + "/*")

    ip = SegProcessing()
    for input_path in input_folder:

        print("+ input_path: ", input_path)
        new_label_nifti = ip.label_overlap(input_path, sequenceOn = [1, 2], sequenceOff = [0])

        ## Get file name, build and write the file name
        file_name = input_path.split(os.path.sep)[-1]

        ## Renaming the new files
        file_name, _ = file_name.split("-bilung")
        file_name = str(file_name + "-" + label_name + ".nii.gz")
        # print("+ file_name:", file_name)

        new_label_folder = os.path.join(sandbox, output_folder)
        Utils().mkdir(new_label_folder)

        nii_file_path = os.path.join(new_label_folder, str(file_name))
        nib.save(new_label_nifti, nii_file_path)


def extract_slices_dataframe(cts_folder, seg_folder, dataframe, sandbox='sandbox', split='train', task='SNF-lung'):
    """
    Function to slicing the lung segmentation from GT index.
    """

    metadata_full = pd.read_csv(dataframe, sep=',')

    ## Using separate folder for training and test
    if split == 'test':
        metadata = metadata_full.query('split == "test"')
        axial_slices_folder = os.path.join(sandbox, task, str("imagesTs"))
        axial_labels_folder = os.path.join(sandbox, task, str("labelsTs"))
    else:
        metadata = metadata_full.query('split == "train"')
        axial_slices_folder = os.path.join(sandbox, task, str("imagesTr"))
        axial_labels_folder = os.path.join(sandbox, task, str("labelsTr"))

    metadata = metadata.reset_index(drop=True)
    print("++ Metadata {}: {}".format(split, metadata.shape))

    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(axial_slices_folder)
    Utils().mkdir(axial_labels_folder)

    ## Iterate between axial slices
    for row in range(metadata.shape[0]):

        ## Locating the CTs files
        if str(metadata['center'][row]) == 'YALE':
            ct_file_name = str(metadata['id_case'][row] + "_HYK.nii.gz")
            ct_file_path = os.path.join(cts_folder, ct_file_name)
        else:
            ct_file_path = os.path.join(cts_folder, metadata['ct_file_name'][row])

        print("+ ct_file_path: ", ct_file_path)


        ## Locating the segmentation files
        if str(metadata['center'][row]) == 'YALE':
            seg_file_name = str(metadata['id_case'][row] + "_HYK-" + task + ".nii.gz")
            seg_file_path = os.path.join(seg_folder, seg_file_name)
        else:
            seg_file_name = str(metadata['id_case'][row] + "_SK-" + task + ".nii.gz")
            seg_file_path = os.path.join(seg_folder, seg_file_name)
            # pass


        ## Fix the position of the slice and check for lesion
        axial_index = metadata['slice_position'][row]


        #########################################################################
        ct_nifti = ImageProcessing().extract_3D_slices(ct_file_path, axial_index)
        seg_nifti = ImageProcessing().extract_3D_slices(seg_file_path, axial_index)

        ## Set the file name for each axial slice and add the mode _0000 for SK
        ct_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '_0000.nii.gz'
        seg_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '-GT_'+ task +'.nii.gz'

        # ## nibabel -> Write axial slices to a Nifti file for each axial slice
        # nib.save(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))
        # nib.save(lesion_nifti, os.path.join(axial_labels_folder, lesion_slice_name))

        ## SimpleITK Write axial slices to a Nifti file for each axial slice
        sitk.WriteImage(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))
        sitk.WriteImage(seg_nifti, os.path.join(axial_labels_folder, seg_slice_name))


def pyRadiomics_feature_extraction(dataframe,
                                    sandbox,
                                    task='SNF-lung',
                                    split="test",
                                    testbed_name="testbed-FirstPaper",
                                    radiomics_set="bb"):
    """ Parallel pyradiomics feature extraction """

    metadata_full = pd.read_csv(dataframe, sep=',')

    ## Using separate folder for training and test
    if split == 'test':
        metadata = metadata_full.query('split == "test"')
        axial_slices_folder = os.path.join(sandbox, task, str("imagesTs"))
        axial_labels_folder = os.path.join(sandbox, task, str("labelsTs"))
    else:
        metadata = metadata_full.query('split == "train"')
        axial_slices_folder = os.path.join(sandbox, task, str("imagesTr"))
        axial_labels_folder = os.path.join(sandbox, task, str("labelsTr"))

    metadata = metadata.reset_index(drop=True)
    print("++ Metadata {}: {}".format(split, metadata.shape))


    ## Flag to write the header of each file
    write_header = False

    ## Crete new folder for feature extraction
    radiomics_folder = os.path.join(sandbox, "testbed", testbed_name, "00_radiomics_features")
    Utils().mkdir(radiomics_folder)


    ## Set file name to write a features vector per case
    filename = str(radiomics_folder+"/00_lesion_features-"+ radiomics_set +".csv")
    features_file = open(filename, 'w+')

    ## Iterate between axial slices
    for row in range(metadata.shape[0]):

        ## Locating the CTs files
        ct_file_name = str(metadata['id_case'][row] + "-" + str(metadata['slice_position'][row]) + "_0000.nii.gz")
        ct_file_path = os.path.join(axial_slices_folder, ct_file_name)
        print("+ ct_file_path: ", ct_file_path)


        ## Locating the segmentation files
        seg_file_name = str(metadata['id_case'][row] + "-" + str(metadata['slice_position'][row]) + "-GT_SNF-lung.nii.gz")
        seg_file_path = os.path.join(axial_labels_folder, seg_file_name)
        print("+ seg_file_path: ", seg_file_path)

        ## Extracting pyradiomics features
        re = RadiomicsExtractor(1)
        feature_extraction_list, image_header_list = re.parallel_extractor(ct_file_path,
                                                                            seg_file_path,
                                                                            str(metadata['id_case'][row] + "-" + str(metadata['slice_position'][row])),
                                                                            metadata['who_label_56-cases'][row],
                                                                            radiomics_set)
        ## writing features by image
        csvw = csv.writer(features_file)
        if write_header == False:
            csvw.writerow(image_header_list)
            write_header = True
        csvw.writerow(feature_extraction_list)

    write_header = False



def pyRadiomics_feature_extraction_Manual(dataframe,
                                    sandbox,
                                    task='SNF-lung',
                                    split="test",
                                    testbed_name="testbed-FirstPaper",
                                    radiomics_set="bb"):
    """ Parallel pyradiomics feature extraction """

    metadata_full = pd.read_csv(dataframe, sep=',')

    ## Using separate folder for training and test
    if split == 'test':
        metadata = metadata_full.query('split == "test"')
        axial_slices_folder = os.path.join(sandbox, task, str("imagesTs"))
        axial_labels_folder = os.path.join(sandbox, task, str("labelsTs"))
    else:
        metadata = metadata_full.query('split == "train"')
        axial_slices_folder = os.path.join(sandbox, task, str("imagesTr"))
        axial_labels_folder = os.path.join(sandbox, task, str("labelsTr"))

    metadata = metadata.reset_index(drop=True)
    print("++ Metadata {}: {}".format(split, metadata.shape))


    ## Flag to write the header of each file
    write_header = False

    ## Crete new folder for feature extraction
    radiomics_folder = os.path.join(sandbox, "testbed", testbed_name, "00_radiomics_features")
    Utils().mkdir(radiomics_folder)


    ## Set file name to write a features vector per case
    filename = str(radiomics_folder+"/00_lesion_features-"+ radiomics_set +".csv")
    features_file = open(filename, 'w+')

    ## Iterate between axial slices
    for row in range(metadata.shape[0]):

        ## Locating the CTs files
        ct_file_name = str(metadata['id_case'][row] + "-" + str(metadata['slice_position'][row]) + "_0000.nii.gz")
        ct_file_path = os.path.join(axial_slices_folder, ct_file_name)
        print("+ ct_file_path: ", ct_file_path)


        ## Locating the segmentation files
        seg_file_name = str(metadata['id_case'][row] + "-" + str(metadata['slice_position'][row]) + "-GT_SNF-lung.nii.gz")
        seg_file_path = os.path.join(axial_labels_folder, seg_file_name)
        print("+ seg_file_path: ", seg_file_path)

        ## Extracting pyradiomics features
        re = RadiomicsExtractor(1)
        feature_extraction_list, image_header_list = re.parallel_extractor(ct_file_path,
                                                                            seg_file_path,
                                                                            str(metadata['id_case'][row] + "-" + str(metadata['slice_position'][row])),
                                                                            metadata['who_label_56-cases'][row],
                                                                            radiomics_set)
        ## writing features by image
        csvw = csv.writer(features_file)
        if write_header == False:
            csvw.writerow(image_header_list)
            write_header = True
        csvw.writerow(feature_extraction_list)

    write_header = False


def run(args):
    """
    Run the pipeline to extract the pyradiomics features from lung segmentations
    """

    ## Step-1: Overlay the lung segmentation clases.
    # folder_3D_label_overlap(args.seg_folder, args.sandbox, label_name="SNF-lung", output_folder="00_GT_Lung_Seg")

    ## Step-2: Slicing the lung segmentation from GT index.
    ## Train
    # extract_slices_dataframe(args.cts_folder,
    #                             args.relabel_seg_folder,
    #                             args.dataframe,
    #                             args.sandbox,
    #                             split="train",
    #                             args.task)

    ## Test
    # extract_slices_dataframe(args.cts_folder,
    #                             args.relabel_seg_folder,
    #                             args.dataframe,
    #                             args.sandbox,
    #                             split="test",
    #                             args.task)

    ## Step-3: Extract features per segmentation layer between all cases
    # pyRadiomics_feature_extraction(args.dataframe,
    #                                 args.sandbox,
    #                                 args.task,
    #                                 split="test",
    #                                 testbed_name=args.task,
    #                                 radiomics_set="bb")

    pyRadiomics_feature_extraction_Manual()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cts', '--cts_folder', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/01_Nifti_Data/')
    parser.add_argument('-seg', '--seg_folder', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/02_GT_Bilung_Seg/')
    parser.add_argument('-rseg', '--relabel_seg_folder', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/sandbox/00_GT_Lung_Seg/')
    parser.add_argument('-d', '--dataframe', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/00_dataframe_sources/56-cases_dataframe_slices-20220419.csv')
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/sandbox/')
    parser.add_argument('-t', '--task', default='SNF-lung')
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
