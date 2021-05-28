
import sys, os
import glob
import shutil

from engine.utils import Utils
from engine.segmentations import LungSegmentations


#######################################################################
## Workflow Launcher settings
#######################################################################

def rotate_CT(dcm_folder_input, dcm_folder_output):
    for slice in dcm_folder_input:
        ## Get slice and rotate position
        slice_name = slice.split(os.path.sep)[-1]
        slice_number, _ = slice_name.split('.')
        rotate_slice = str(-(int(slice_number) - 350))

        ## Set source file
        dir_input = str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/multiomics/01_segmentation_test/full-body2chest/CT_source/TY113/")
        input_file_path = os.path.join(dir_input, slice_name)

        ## Set new file name to write the new slice position
        output_file_path = os.path.join(dcm_folder_output, str(rotate_slice + '.dcm'))

        ## Copy the slice in a roate position
        shutil.copyfile(input_file_path, output_file_path)

        print("======================================")
        print("+ output file_path", output_file_path)
        print("+ input file_path", input_file_path)

# #######################################################################
# ## Rotate DICOM CT scan series
# dcm_folder_input = sorted(glob.glob(str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/multiomics/01_segmentation_test/full-body2chest/CT_source/TY113/*")))
# dcm_folder_output = str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/multiomics/01_segmentation_test/full-body2chest/CT_source/TY113-rotate/")



######################################################################
## Convert DICOM CT scans to Nifti
# testbed = "testbed/"
# dcm_folder = glob.glob(str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Parma/04_05_cases-20210518/ParmaDataDicom/*"))
# nii_folder = str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Parma/04_05_cases-20210518/ParmaDataNifti/")
# ##Utils().convert_dcm2nii(dcm_folder, nii_folder)
# Utils().itk_convert_dcm2nii(dcm_folder, nii_folder)


#######################################################################
## CT lung lobes segmentation
##input_folder = glob.glob(str(testbed + "/dataset_unibe/train-nii/*"))
nii_folder = str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/06_11_cases-20210503/BernDataNifti-1/*")
input_folder = glob.glob(nii_folder)
output_folder = str("//data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/06_11_cases-20210503/BernDataLungSeg/")

ls = LungSegmentations()
for input_path in input_folder[:]:
    ## Formatting the folder for each patient case
    input_case = glob.glob(str(input_path + "/*"))
    # input_case = input_path
    folder_name = input_path.split(os.path.sep)[-1]
    output_case = os.path.join(output_folder, folder_name)
    Utils().mkdir(output_case)
    # print("input_case: ", input_case)
    # print("input_case: ", input_path)
    ls.folder_segmentations(input_case, output_case, 'bi-lung', 5)
    ## ls.folder_segmentations(input_case, output_case, 'lobes', 5)


# #######################################################################
# ## Convert nrrd to nifti file
# nrrd_file = "/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/06_11_cases-20210503/BernDataNifti/B0031_06_200609_CT_SK.nrrd"
# nii_file = "/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/06_11_cases-20210503/BernDataNifti/B0031_06_200609_CT_SK.nii"
# img_ = Utils().read_nrrd(nrrd_file)
# Utils().rwrite_nifti(img_, nii_file)
