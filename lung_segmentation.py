
import argparse
import json
import logging

import sys, os
import glob
import shutil
import SimpleITK as sitk

from engine.utils import Utils
from engine.segmentations import LungSegmentations


def run(args):
    nii_folder = args.input
    input_folder = glob.glob(nii_folder + "/*")
    output_folder = args.output
    nii_folder = args.nii_folder

    ls = LungSegmentations()
    for input_path in input_folder:

        ## Get folder name
        folder_name = input_path.split(os.path.sep)[-1]

        ## Read dicom image series and write it in a Nifti formt
        dicom_img =  Utils().read_dicom(input_path)
        if nii_folder:
            ## Write CT scan in a Nifti format
            Utils().mkdir(nii_folder)
            nii_file_name = os.path.join(nii_folder, str(folder_name + ".nii.gz"))
            sitk.WriteImage(dicom_img, nii_file_name)


        ## Generating the lung segmentation and write it in a Nifti Formatting
        lung_segmentation = ls.ct_segmentation(dicom_img, args.seg_method, args.batch)

        Utils().mkdir(output_folder)
        lung_file_name = os.path.join(output_folder, str(folder_name + "-bi-lung.nii.gz"))
        sitk.WriteImage(lung_segmentation, lung_file_name)
        print("CT segmentation file: {}".format(str(lung_file_name)))


def main():
    """
    This function is used to segment the lung from the input CT scan.
    + The input CT scan is a Dicom image series.
    + The output lung segmentation is a Nifti image stored in the output folder,
        and the file name is the same as the input CT scan file name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/data/01_UB/01_Multiomics-Data/Clinical_Imaging/05_Step-4_CasesSourcesYale/01_Dicom_Sources')
    parser.add_argument('-nii', '--nii_folder', default='/data/01_UB/01_Multiomics-Data/Clinical_Imaging/05_Step-4_CasesSourcesYale//02_Nifti_Data')
    parser.add_argument('-o', '--output', default='//data/01_UB/01_Multiomics-Data/Clinical_Imaging/05_Step-4_CasesSourcesYale/03_Nifti_Vienna_LungSegmentation')
    parser.add_argument('-s', '--seg_method', type=str, default='bi-lung')
    parser.add_argument('-b', '--batch', type=int, default=5)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
