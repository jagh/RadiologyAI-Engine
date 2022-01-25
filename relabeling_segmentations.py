
import logging
import argparse

import sys, os
import glob
import nibabel as nib

from engine.utils import Utils
from engine.preprocessing import SegProcessing


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

    ip = SegProcessing()
    for input_path in input_folder:

        relabel_lesion_nifti = ip.relabel_sequential(input_path, sequence = [0, 1, 2, 4, 5, 6])
        # print("relabel_lesion_nifti: ", relabel_lesion_nifti.shape)

        ## Get file name, build and write the file name
        file_name = input_path.split(os.path.sep)[-1]

        relabel_folder = os.path.join(sandbox, str("00-relabel_folder"))
        Utils().mkdir(relabel_folder)

        nii_file_path = os.path.join(relabel_folder, str(file_name))
        nib.save(relabel_lesion_nifti, nii_file_path)



def folder_3D_label_overlap(input_folder, sandbox, label_name="SNF_General"):
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

    ip = SegProcessing()
    for input_path in input_folder:
        new_label_nifti = ip.label_overlap(input_path, sequence = [1, 2, 3, 4, 5])
        # print("relabel_lesion_nifti: ", relabel_lesion_nifti.shape)

        ## Get file name, build and write the file name
        file_name = input_path.split(os.path.sep)[-1]

        ## Renaming the new files
        file_name, _ = file_name.split("-SNF_bilungs")
        file_name = str(file_name + "-" + label_name + ".nii.gz")
        # print("+ file_name:", file_name)

        new_label_folder = os.path.join(sandbox, str("00-relabel_folder"))
        Utils().mkdir(new_label_folder)

        nii_file_path = os.path.join(new_label_folder, str(file_name))
        nib.save(new_label_nifti, nii_file_path)



def run(args):
    input_folder = glob.glob(args.input + "/*")
    # folder_relabel_segmentations(input_folder, args.sandbox)

    folder_3D_label_overlap(input_folder, args.sandbox, args.label_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/04_included_cases_FP/00_Nifti-3D_Lesion/')
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/04_included_cases_FP/sandbox/')
    parser.add_argument('-n', '--label_name', default="SNF_General")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
