"""
Disease severity classification of lung tissue abnormalities using the MosMedata.
"""

import sys, os
import glob

from engine.utils import Utils
from engine.segmentations import LungSegmentations


def auto_segmentation(studies_folder, segmentation_folder, seg_method):
    """ Formatting the folder studies for each GGO categories """
    for std_path in studies_folder:
        input_std = glob.glob(str(std_path + "/*"))
        folder_name = std_path.split(os.path.sep)[-1]
        output_std = os.path.join(segmentation_folder, folder_name)
        Utils().mkdir(output_std)

        ## Launch of automatic CT segmentation
        LungSegmentations().folder_segmentations(input_std, output_std, seg_method, 5)



#######################################################################
## Workflow Launcher settings
#######################################################################

## Dataset path definitions
studies_folder = glob.glob("/data/01_UB/CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/*")
testbed = "testbed/"
lung_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_bi-lung/")
lobes_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_lobes/")

## Stage-1: CT bi-lung and lung lobes segmentation
auto_segmentation(studies_folder, lung_segmentation_folder, 'bi-lung')
auto_segmentation(studies_folder, lobes_segmentation_folder, 'lobes')
