
import sys, os
import glob

from engine.utils import Utils
from engine.segmentations import LungSegmentations


#######################################################################
## Workflow Launcher settings
#######################################################################

## Convert CT scans
testbed = "testbed/"
dcm_folder = glob.glob(str("/data/01_UB/CT_Datasets/dataset_UniBern/sources/*"))
nii_folder = str(testbed + "/dataset_unibe/train-nii/")

Utils().convert_dcm2nii(dcm_folder, nii_folder)



# #######################################################################
# ## CT lung lobes segmentation
# input_folder = glob.glob(str(testbed + "/dataset_unibe/train-nii/*"))
# output_folder = str(testbed + "/dataset_unibe/segmentations_lobes/")
#
# ls = LungSegmentations()
# for input_path in input_folder[:2]:
#     ## Formatting the folder for each patient case
#     input_case = glob.glob(str(input_path + "/*"))
#     folder_name = input_path.split(os.path.sep)[-1]
#     output_case = os.path.join(output_folder, folder_name)
#     Utils().mkdir(output_case)
#
#     ls.folder_segmentations(input_case, output_case, 'lobes', 5)
