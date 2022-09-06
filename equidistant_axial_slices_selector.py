

import argparse
from ctypes import util
import json
import logging

import sys, os
import glob
import nibabel as nib

import pandas as pd
import numpy as np
import csv

from engine.utils import Utils
from engine.preprocessing import ImageProcessing
from engine.preprocessing import SegProcessing


def equidistant_axial_slices_selector(sandbox, lung_folder, output_csv):
    """
    Method to select ten equidistant slices per CT scan from the 3D lung segmentations.
    """

    input_folder = glob.glob(sandbox + "/06_2nd-Nifti_Data/*")

    sp = SegProcessing()
    for input_path in input_folder:

        ## Get file name
        file_name = input_path.split(os.path.sep)[-1]
        file_name, _ = file_name.split(".nii.gz")
        print("+ file_name:", file_name)

        ## Set lung segmentation file path
        lung_file_path = os.path.join(lung_folder, file_name + "-bi-lung.nii.gz")
        # print("+ lung_file_path:", lung_file_path)    

        ## 3D lung scan load
        lung = nib.load(lung_file_path)
        lung_array = lung.get_fdata()

        ## Get the number of slices
        num_slices = lung_array.shape[2]
        # print("+ num_slices:", num_slices)

        ## Get the upper and lower bounds of the lung segmentation
        # upper_bound, lower_bound = sp.get_upper_lower_bounds(lung_array)
        upper_bound = np.where(lung_array == 1)[0].max()
        lower_bound = np.where(lung_array == 1)[0].min()
        # print("+ upper_bound:", upper_bound)
        # print("+ lower_bound:", lower_bound)

        ## Get ten equidistant slices
        num_slices = 12
        # equidistant_slices = sp.get_equidistant_slices(upper_bound, lower_bound, num_slices)
        equidistant_slices = np.linspace(lower_bound, upper_bound, num_slices, dtype=int)
        # print("+ equidistant_slices:", equidistant_slices)

        ## Remove the slices the upper and lower bounds
        equidistant_slices = np.delete(equidistant_slices, [0, -1])
        equidistant_slices = equidistant_slices.T
        print("+ equidistant_slices:", equidistant_slices)

        ## Save a pairwise each equidistant with the file name
        with open(output_csv, 'a') as f:
            writer = csv.writer(f)
            for i in range(len(equidistant_slices)):
                writer.writerow([file_name, equidistant_slices[i]])
        
        
        

        
        



def run(args):
    """
    Run the pipeline to extract the pyradiomics features from lung segmentations
    """

    # Step-1: Overlay the lung segmentation clases.
    equidistant_axial_slices_selector(args.sandbox,
                                        args.lung_folder, 
                                        args.output_csv,)


def main():
    ## Args to execute the step 1 and 2
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sandbox', default='//data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/')
    parser.add_argument('-lu', '--lung_folder', default='//data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/06_2nd-testset_LungSeg/')
    parser.add_argument('-o', '--output_csv', default='//data/01_UB/00_Dev/01_SNF_Dataset_First_Paper/equidistant_slices.csv')
    args = parser.parse_args()
    run(args)




if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()