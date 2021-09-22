
import argparse
import json
import logging

import sys, os
import glob
import shutil
import SimpleITK as sitk

from engine.utils import Utils
from engine.segmentations import LungSegmentations

######################################################################
def run(args):
    nii_folder = args.input
    input_folder = glob.glob(nii_folder + "/*")
    output_folder = args.output

    ls = LungSegmentations()
    for input_path in input_folder[:]:
        ## Formatting the folder for each patient case
        ct_name = input_path.split(os.path.sep)[-1]
        ct_dcm_format = str(ct_name.split('.nii.gz')[0] + '-' + args.seg_method + '.nii.gz')

        input_ct = sitk.ReadImage(input_path)
        result_out = ls.ct_segmentation(input_ct, args.seg_method, args.batch)

        Utils().mkdir(output_folder)
        sitk.WriteImage(result_out, str(output_folder+"/"+ct_dcm_format))
        print("CT segmentation file: {}".format(str(output_folder+"/"+ct_dcm_format)))


import pandas as pd
import json

def main():

    ######################################################################
    ######################################################################
    ##
    # object = pd.read_pickle(r'/home/jagh/Documents/01_UB/17_lession_segmentation/05_nnUNet_Analysis/nnunet_issue/plans.pkl')
    # # print('object: ', object)
    # print('object: ', type(object))
    #
    # file_name = '/home/jagh/Documents/01_UB/17_lession_segmentation/05_nnUNet_Analysis/nnunet_issue/plans.txt'
    # with open(file_name, 'w+') as f:
    #     json.dump(object, f, indent=4)
    ######################################################################
    ######################################################################


    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/02_Step-3_109-CasesSegmented/01_Nifti-Data/')
    parser.add_argument('-o', '--output', default='//data/01_UB/Multiomics-Data/Clinical_Imaging/02_Step-3_109-CasesSegmented/03_Nifti-LungSeg/')
    parser.add_argument('-s', '--seg_method', type=str, default='bi-lung')
    parser.add_argument('-b', '--batch', type=int, default=5)

    args = parser.parse_args()
    run(args)


    #
    # with open(object) as json_file:
    #     json_data = json.load(json_file)
    # print(json_data)





if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
