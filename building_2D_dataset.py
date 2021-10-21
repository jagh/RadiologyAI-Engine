
import logging
import argparse

import sys, os
import glob
import nibabel as nib

from engine.utils import Utils
from engine.preprocessing import ImagePreprocessing


def folder_relabel_segmentations(input_folder, sandbox):
    """
    This function is used to relabel the lesion nifti files stred in a folder.

        :param input_path: the path of the input nifti file
        :type input_path: str

        :param sequence: the sequence of the relabel
        :type sequence: list

        :return: the relabeled nifti file
        :rtype: nifti file
    """

    ip = ImagePreprocessing()
    for input_path in input_folder:

        relabel_lesion_nifti = ip.relabel_sequential(input_path, sequence = [0, 1, 2, 4, 5, 6])
        # print("relabel_lesion_nifti: ", relabel_lesion_nifti.shape)

        ## Get file name, build and write the file name
        file_name = input_path.split(os.path.sep)[-1]

        relabel_folder = os.path.join(sandbox, str("00-relabel_folder"))
        Utils().mkdir(relabel_folder)

        nii_file_path = os.path.join(relabel_folder, str(file_name))
        nib.save(relabel_lesion_nifti, nii_file_path)




def run(args):
    nii_folder = args.input
    input_folder = glob.glob(nii_folder + "/*")
    sandbox = args.sandbox

    folder_relabel_segmentations(input_folder, sandbox)




def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/data/01_UB/03_Data_developments/02_Nifti-Seg-6-Classes/')
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/03_Data_developments/sandbox/')
    parser.add_argument('-d', '--dataframe', default='//data/01_UB/03_Data_developments/2_dataframe_axial_slices/')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
