
import logging
import argparse

import sys, os
import glob
import nibabel as nib

from engine.utils import Utils
from engine.preprocessing import ImagePreprocessing


def run(args):
    nii_folder = args.input
    input_folder = glob.glob(nii_folder + "/*")
    output_folder = args.output

    ip = ImagePreprocessing()
    for input_path in input_folder:

        relabel_lesion_nifti = ip.relabel_segmentation(input_path)
        print("relabel_lesion_nifti: ", relabel_lesion_nifti.shape)

        ## Get folder name and write the file name
        folder_name = input_path.split(os.path.sep)[-1]
        nii_file_name = os.path.join(output_folder, str(folder_name))

        nib.save(relabel_lesion_nifti, nii_file_name)





def main():
    """    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/data/01_UB/03_Data_developments/02_Nifti-Seg-6-Classes/')
    parser.add_argument('-o', '--output', default='/data/01_UB/03_Data_developments/03_output_folder/')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
