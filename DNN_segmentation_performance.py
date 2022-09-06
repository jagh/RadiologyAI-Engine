
import argparse
from ctypes import util
import json
import logging

import sys, os
import glob
import shutil
import pandas as pd
import numpy as np
import csv

import nibabel as nib
import SimpleITK as sitk

import matplotlib.pyplot as plt

from engine.utils import Utils
from engine.segmentations import LungSegmentations
from engine.featureextractor import RadiomicsExtractor
from engine.medical_image_metrics import MIM


def dnn_segmentation_performance(sandbox, cts_folder, gt_folder, dnn_folder, dataframe, output_file_name):
    """
    Compare the DNN segmentation with the GT segmentation from a dataframe.
    """

    ## Read the dataframe
    metadata = pd.read_csv(dataframe)
    print("+ metadata:", metadata.shape)

    ## Output file
    mim_file = open(output_file_name, 'w+')
    write_header = False

    ## iterate over the dataframe
    # for index, row in metadata.iterrows():
    for index, row in metadata[:30].iterrows():
        # print("+ index:", index)
        # print("+ row:", row)

        print("++ row", row['study_ID_split'], " || ", row['slice_index'])

        ## Locate the GT segmentation and DNN segmentation
        if row['center'] == 'YALE':
            gt_file_path = os.path.join(gt_folder, row['study_ID_split'] + "_HYK-gtlesion.nii.gz")
            dnn_file_path = os.path.join(dnn_folder, row['study_ID_split'] + "_HYK-SNF_bilungs.nii.gz")
            print("++ gt_file_path:", gt_file_path)
            print("++ dnn_file_path:", dnn_file_path)
        else:
            gt_file_path = os.path.join(gt_folder, row['study_ID_split'] + "_SK-gtlesion.nii.gz")
            dnn_file_path = os.path.join(dnn_folder, row['study_ID_split'] + "_SK-SNF_bilungs.nii.gz")
            print("++ gt_file_path:", gt_file_path)
            print("++ dnn_file_path:", dnn_file_path)

        ## Read the GT segmentation and DNN segmentation
        gt_nii = nib.load(gt_file_path)
        gt_data = gt_nii.get_fdata()

        dnn_nii = nib.load(dnn_file_path)
        dnn_data = dnn_nii.get_fdata()

        ## Extract the slice from the GT segmentation and DNN segmentation
        gt_slice = gt_data[:,:,row['slice_index']]
        dnn_slice = dnn_data[:,:,row['slice_index']]
        print("++ gt_slice:", gt_slice.shape)
        print("++ dnn_slice:", dnn_slice.shape)

        ## Compute the metrics
        mim = MIM()
        mim_values_list, mim_header_list = mim.binary_metrics(dnn_slice, gt_slice, row['study_ID_split'], row['slice_index'], row['WHO_score day_0'])
        print("++ mim_values_list:", mim_values_list)
        print("++ mim_header_list:", mim_header_list)

        ## writing features by image
        csvw = csv.writer(mim_file, delimiter=',')
        if write_header == False:
            csvw.writerow(mim_header_list)
            write_header = True
        csvw.writerow(mim_values_list)




def set_segmentation_class(segmentation_array, sequenceOn, sequenceOff):
    """
    Set the segmentation classes.
    """
    ## Set the segmentation class
    try:
        ## new lesion array
        new_segmentation = np.zeros_like(segmentation_array)

        ## Change the  sequence on
        for seq in sequenceOn:
            new_segmentation[segmentation_array == seq] = 1

        ## Remove the sequence off
        for seq in sequenceOff:
            new_segmentation[segmentation_array == seq] = 0

    except(Exception, ValueError) as e:
        print("Not lesion segmentation")

    segmentation_classes = {
        'background': 0,
        'GGO': 1,
        'CON': 2,
        'PLE': 3,
        'BAN': 4,
        'TBR': 5,
        }

    ## Get the segmentation class from the sequenceOn
    for key, value in segmentation_classes.items():
        if value == sequenceOn[0]:
            segmentation_class = key

    return new_segmentation, segmentation_class



def dnn_segmentation_performace_by_class(sandbox, cts_folder, gt_folder, dnn_folder, dataframe, output_file_name):
    """ 
    Compare the DNN segmentation with the GT segmentation from a dataframe by segmentation class.
    """

    ## Read the dataframe
    metadata = pd.read_csv(dataframe)
    print("+ metadata:", metadata.shape)

    ## Output file
    mim_file = open(output_file_name, 'w+')
    write_header = False

    ## iterate over the dataframe
    # for index, row in metadata.iterrows():
    for index, row in metadata[:30].iterrows():
        # print("+ index:", index)
        # print("+ row:", row)

        print("++ row", row['study_ID_split'], " || ", row['slice_index'])

        ## Locate the GT segmentation and DNN segmentation
        if row['center'] == 'YALE':
            gt_file_path = os.path.join(gt_folder, row['study_ID_split'] + "_HYK-gtlesion.nii.gz")
            dnn_file_path = os.path.join(dnn_folder, row['study_ID_split'] + "_HYK-SNF_bilungs.nii.gz")
            print("++ gt_file_path:", gt_file_path)
            print("++ dnn_file_path:", dnn_file_path)
        else:
            gt_file_path = os.path.join(gt_folder, row['study_ID_split'] + "_SK-gtlesion.nii.gz")
            dnn_file_path = os.path.join(dnn_folder, row['study_ID_split'] + "_SK-SNF_bilungs.nii.gz")
            print("++ gt_file_path:", gt_file_path)
            print("++ dnn_file_path:", dnn_file_path)


        ## Read the GT segmentation and DNN segmentation
        gt_nii = nib.load(gt_file_path)
        gt_data = gt_nii.get_fdata()

        dnn_nii = nib.load(dnn_file_path)
        dnn_data = dnn_nii.get_fdata()

    

        ## Extract the slice from the GT segmentation and DNN segmentation
        sequence = [1, 2, 3, 4, 5]
        for i in sequence:
            ## drop the current sequence
            sequenceOff = np.delete(sequence, i-1)
            sequenceOff = np.append(sequenceOff, 0)
            # print("++ secuenceOff:", sequenceOff)

            sequenceOn = [i,]
            # print("++ secuenceOn:", sequenceOn)

            ## Set the segmentation class                 
            gt_new_data, seg_class = set_segmentation_class(gt_data, sequenceOn, sequenceOff)
            dnn_new_data, _ = set_segmentation_class(dnn_data, sequenceOn, sequenceOff)
            
            ## Extract the slice from the GT segmentation and DNN segmentation
            gt_slice = gt_new_data[:,:,row['slice_index']]
            dnn_slice = dnn_new_data[:,:,row['slice_index']]
            print("++ gt_slice:", gt_slice.shape)
            print("++ dnn_slice:", dnn_slice.shape)

            ## Compute the metrics
            mim = MIM()
            mim_values_list, mim_header_list = mim.binary_metrics(dnn_slice, gt_slice, row['study_ID_split'], row['slice_index'], row['WHO_score day_0'], seg_class)
            print("++ mim_values_list:", mim_values_list)
            print("++ mim_header_list:", mim_header_list)

            ## writing features by image
            csvw = csv.writer(mim_file, delimiter=',')
            if write_header == False:
                csvw.writerow(mim_header_list)
                write_header = True
            csvw.writerow(mim_values_list)





def run(args):
    """
    Run the pipeline to extract the pyradiomics features from lung segmentations
    """

    # Step-1: Overlay the lung segmentation clases.
    # dnn_segmentation_performance(args.sandbox,
    #                             args.cts_folder, 
    #                             args.gt_folder,
    #                             args.dnn_folder,
    #                             args.dataframe,
    #                             args.output_file_name)


    dnn_segmentation_performace_by_class(args.sandbox,
                                args.cts_folder, 
                                args.gt_folder,
                                args.dnn_folder,
                                args.dataframe,
                                args.output_file_name)




def main():
    ## Args to execute the step 1 and 2
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/01_Nifti_Data/')
    parser.add_argument('-cts', '--cts_folder', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/01_GT_Lung_Seg/')
    parser.add_argument('-gt', '--gt_folder', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/01_GT_Lesion_Seg/')
    parser.add_argument('-dnn', '--dnn_folder', default='/data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/01_DNN_Lesion_Seg_Raw/')
    parser.add_argument('-df', '--dataframe', default='/data/01_UB/00_Dev/02_Dev_Pipeline_Execution/AssessNet-19_RAI-Dataset/02_dataframe_sources_3V/05_axial_slices/dataframeTs-270_subjects-RAI-20220824.csv')
    parser.add_argument('-o', '--output_file_name', default='/home/jagh/Documents/01_UB/RadiologyAI-Engine/testbed-segmentation_performance/multi-lesion_segmentation.csv')
    args = parser.parse_args()
    run(args)




if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()