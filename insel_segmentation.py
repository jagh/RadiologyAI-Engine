
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
# dcm_folder = glob.glob(str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/03_23cases-20210325/YaleDataDicom/*"))
# nii_folder = str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/03_23cases-20210325/YaleDataNifti/")
# Utils().convert_dcm2nii(dcm_folder, nii_folder)





# #######################################################################
# ## CT lung lobes segmentation
# # input_folder = glob.glob(str(testbed + "/dataset_unibe/train-nii/*"))
# nii_folder = str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/03_23cases-20210325/YaleDataNifti/*")
# input_folder = glob.glob(nii_folder)
# output_folder = str("/data/01_UB/Multiomics-Data/Clinical_Imaging/Yale/03_23cases-20210325/YaleDataLungSeg/")
#
# ls = LungSegmentations()
# for input_path in input_folder[:]:
#     ## Formatting the folder for each patient case
#     input_case = glob.glob(str(input_path + "/*"))
#     # input_case = input_path
#     folder_name = input_path.split(os.path.sep)[-1]
#     output_case = os.path.join(output_folder, folder_name)
#     Utils().mkdir(output_case)
#
#
#     print("input_case: ", input_case)
#     print("input_case: ", input_path)
#
#     ls.folder_segmentations(input_case, output_case, 'bi-lung', 5)
# #     # ls.folder_segmentations(input_case, output_case, 'lobes', 5)







##################################################################################
## Example 1: https://github.com/faustomilletari/VNet/blob/master/utilities.py
## Example 2: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/31_Levelset_Segmentation.html
## Exmples : https://www.programcreek.com/python/example/123390/SimpleITK.ResampleImageFilter
## SITK Doc: https://simpleitk.readthedocs.io/en/master/IO.html
## ITK Doc: https://itkpythonpackage.readthedocs.io/en/master/Quick_start_guide.html

# import SimpleITK as sitk
# from radiomics import featureextractor, firstorder, shape
#
# print(sitk.Version())
#
#
# img_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft.nii'
# seg_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft.niibilung.nii'
#
# ## Red the images
# img = sitk.ReadImage(img_path)
# seg = sitk.ReadImage(seg_path)
#
# print('+ img:', type(img))
# print('+ seg:', type(seg))
# print('+ seg:', seg)


# ## Common error -1:
# ##   TypeError: Execute() missing 1 required positional argument: 'featureImage'
# sitkLI2LM = sitk.LabelMapToLabelImageFilter()
# labelmaps = sitkLI2LM.Execute(seg)
# print("labelmaps", labelmaps)


##################################################################################
## Nibabel opt 1: Making and saving new images in nibabel
## https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
## Nibabel opt 2: https://github.bajins.com/nipy/nilabels

import numpy as np
import nibabel as nib
import nilabels as nil


# img_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft.nii'
# seg_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft.niibilung.nii'
# new_seg_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft.relabel.nii'
#
# slice_seg_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft.slice.nii'
# slice_relabel_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft.slice.relabel.nii'
#
# # nil_app = nil.App()
# # nil_app.manipulate_labels.relabel(seg_path, new_seg_path,  [1, 2, 3, 4], [4, 3, 2, 1])
#
#
# ## Red the image and segmentation
# img = nib.load(img_path)
# img_data = img.get_data()
# img_affine = img.affine
#
# seg = nib.load(seg_path)
# seg_data = seg.get_data()
# seg_affine = seg.affine
#
# print('seg_data: ', seg_data.shape)
# print('seg_data: ', seg_data.shape[2])
#
# ## Get slice and write
# slice_seg_data = seg_data[:,:,169]
# nft_seg = nib.Nifti1Image(slice_seg_data, seg_affine)
# nib.save(nft_seg, slice_seg_path)
#
# nil_app = nil.App()
# nil_app.manipulate_labels.relabel(slice_seg_path, slice_relabel_path,  [1, 2], [5, 6])
#
# print('slice_seg_data: ', slice_seg_data)




##########################################################
## Working

import numpy as np
import nibabel as nib
import glob

# def merge_nii_files (sfile, ns):
#     # This will load the first image for header information
#     # img = ni.load(sfile % (3, ns[0]))
#     img = nib.load(sfile)
#     dshape = list(img.shape)
#     dshape.append(len(ns))
#     data = np.empty(dshape, dtype=img.get_data_dtype())
#
#     header = img.header
#     equal_header_test = True
#
#     # Now load all the rest of the images
#     for n, i in enumerate(ns):
#         img = ni.load(sfile % (3,i))
#         equal_header_test = equal_header_test and img.header == header
#         data[...,n] = np.array(img.dataobj)
#
#     imgs = ni.Nifti1Image(data, img.affine, header=header)
#     if not equal_header_test:
#         print("WARNING: Not all headers were equal!")
#     return(imgs)
#
#
# nii_files = glob.glob("/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/merge/COVID_19_Testcase1_1_soft*")
# images = merge_nii_files(nii_files, range(1,2))



seg_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/merge/COVID_19_Testcase1_1_soft-bilung.nii'
slice_relabel_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/merge/COVID_19_Testcase1_1_soft.relabel.nii'
segs_path = '/data/01_UB/Multiomics-Data/Clinical_Imaging/Bern/Test/COVID_19_Testcase1_1_soft-segs.relabel.nii'
ns = range(1,2)


seg = nib.load(seg_path)
dshape = list(seg.shape)
dshape.append(len(ns))
data = np.empty(dshape, dtype=seg.get_data_dtype())

header = seg.header
equal_header_test = True


for n, i in enumerate(ns):
    # Now load all the rest of the images
    seg_2 = nib.load(slice_relabel_path)
    equal_header_test = equal_header_test and seg_2.header == header
    data[...,n] = np.array(seg.dataobj)


segs = nib.Nifti1Image(data, seg.affine, header=header)
if not equal_header_test:
    print("WARNING: Not all headers were equal!")

nib.save(segs, segs_path)








# img_orientation = nib.aff2axcodes(img_affine)
# seg_orientation = nib.aff2axcodes(seg_affine)
#
# print('img_orientation', img_orientation)
# print('seg_orientation', seg_orientation)





#stikLOIF = sitk.LabelOverlayImageFilter()
# stikLOIF.Execute(img, seg)
#values = stikLOIF.GetBackgroundValue(img, seg)

# print('values: ', values)

#
# seg_ = sitk.BinaryDilate(seg, 3)
# print("seg_: ", seg_)


#
# ## Get images from array
# sitk_img = sitk.GetImageFromArray(img, isVector=False)
# sitk_seg = sitk.GetImageFromArray(seg, isVector=False)
#
# print('sitk_img', sitk_img)
# print('sitk_seg', sitk_seg)
