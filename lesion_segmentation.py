
import logging
import argparse

import sys, os
import glob
import nibabel as nib

import pandas as pd
import numpy as np

from engine.utils import Utils
from engine.preprocessing import ImageProcessing


def convert_axial_slices(dataframe, cts_folder, seg_folder, sandbox='sandbox', split='train'):
    """
    This function converts the dataset into a format that is compatible with the nnUNet framework.
    The function takes as input the metadata file, the folder containing the CTs, and
    the folder containing the manual segmentations.
    Finally, the function iterates over the metadata file and creates a Nifit file
    for each axial slice with shape (x, y, 1).

    Parameters
    ----------
        dataframe (str): The metadata file.
        cts_folder (str): The folder containing the CTs.
        seg_folder (str): The folder containing the GT.
        sandbox (str): Output folder.
        split (str): train or test.
    Returns:
    -------
        None
    """

    metadata_full = pd.read_csv(dataframe, sep=',')

    ## Using separate folder for training and test
    if split == 'test':
        metadata = metadata_full.query('split == "test"')
        axial_slices_folder = os.path.join(sandbox, str("All_axial_imagesTs"))
        # axial_labels_folder = os.path.join(sandbox, str("axial_labelsTs"))
    else:
        metadata = metadata_full.query('split == "train"')
        axial_slices_folder = os.path.join(sandbox, str("All_axial_imagesTr"))
        # axial_labels_folder = os.path.join(sandbox, str("labelsTr"))

    ## Remove axial slices rows and resert index
    metadata = metadata.drop_duplicates(subset=['id_case'])
    metadata = metadata.reset_index(drop=True)
    print("++ Metadata {}: {}".format(split, metadata.shape))

    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(axial_slices_folder)
    # Utils().mkdir(axial_labels_folder)

    ## Iterate between axial slices
    for row in range(metadata.shape[0]):
        ## Locating the CTs and labels
        ct_file_name = os.path.join(cts_folder, metadata['ct_file_name'][row])
        print("+ ct_file_name: ", ct_file_name)

        ## get the ct array
        image = nib.load(ct_file_name)
        image_array = image.get_fdata()

        ## Loop into all axial slices by CT scan
        for axial_index in range(image_array.shape[2]):

            ct_nifti = ImageProcessing().extract_axial_slice_3D(ct_file_name, axial_index)

            ## Set the file name for each axial slice and add the mode _0000 for SK
            ct_slice_name = str(metadata['id_case'][row]) + '-' + str(axial_index) + '_0000.nii.gz'

            ## Write axial slices to a Nifti file for each axial slice
            nib.save(ct_nifti, os.path.join(axial_slices_folder, ct_slice_name))


def build_3D_images(split='train', task='bi-lung'):

    ## Testbed path
    data_testbed_root = "/data/01_UB/01_Multiomics-Data/Clinical_Imaging/02_Step-3_122-CasesSegmented/"

    ## Dataset path definitions
    nifti_folder = os.path.join(data_testbed_root, "01_Nifti-Data/")
    # lesion_folder = os.path.join(data_testbed_root, "02_Nifti-Seg-6-Classes/")
    lesion_folder = os.path.join(data_testbed_root, "03_Nifti-LungSeg/")

    ## Metadat path
    metadata_path = "/home/jagh/Documents/01_UB/10_Conferences_submitted/08_MIA-Explainable_2021/02_materials/dataset/"
    metadata_file_path = os.path.join(metadata_path, "1XX_dataframe_axial_slices-Temp.csv")
    axial_metadata_file_path= os.path.join(metadata_path, "1XX_dataframe_axial_slices-Temp.csv")

    ## Read the metadata
    metadata_full = pd.read_csv(metadata_file_path, sep=',')
    print("+ Metadata: ", metadata_full.shape)

    ## nn-UNet sandbox paths
    dataset_folder = '/data/01_UB/00_Dev/nnUNet_sandbox_bi-lungs/nnUNet_raw_data/Task115_COVIDSegChallenge/'

    ## Axial slices folder
    if split == 'test':
        axial_lesion_folder = os.path.join(dataset_folder, "AAxial-labelsTs/")
        labelsTr_shape_path = os.path.join(dataset_folder, "3D-labelsTs")
        metadata = metadata_full.query('split == "test"')
    else:
        axial_lesion_folder = os.path.join(dataset_folder, "AAxial-labelsTr/")
        labelsTr_shape_path = os.path.join(dataset_folder, "3D-labelsTr")
        metadata = metadata_full.query('split == "train"')

    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(labelsTr_shape_path)

    ## Using separate folder for training and test
    metadata = metadata.drop_duplicates(subset=['id_case'])
    metadata = metadata.reset_index(drop=True)
    # print("++ metadata:", metadata.head())
    print("+ Metadata {}: {}".format(split, metadata.shape))


    ## Loop through dataset
    for row in range(metadata.shape[0]):
        # print('+ row: ', row)
        # print('+ ct_file_name: ', metadata['ct_file_name'][row])

        ## locating the CT and Seg
        ct_scan_input_path = os.path.join(nifti_folder, metadata['ct_file_name'][row])

        if task == 'bi-lung':
            lesion_scan_input_path = os.path.join(lesion_folder, metadata['dnn_lung_file_name'][row])
        else:
            lesion_scan_input_path = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

        # print('+ ct_scan_input_path', ct_scan_input_path)
        # print('+ lesion_scan_input_path', lesion_scan_input_path)

        ## Reads .nii file and get the numpy image array
        ct = nib.load(ct_scan_input_path)
        ct_scan_array = ct.get_data()
        # print('+ ct_scan_array', ct.header)

        lesion = nib.load(lesion_scan_input_path)
        lesion_scan_array = lesion.get_data()
        # print("lesion_scan_array: ", lesion_scan_array.shape)

        #####################################################################
        #####################################################################
        ## New lesion segmentatio
        new_3D_lesion_array = np.zeros_like(lesion_scan_array)
        # print("new_3D_lesion_array: ", new_3D_lesion_array.shape)

        for slice in range(lesion_scan_array.shape[2]):

            slice_position = slice

            axial_lesion_filename, _ = str(metadata['lesion_file_name'][row]).split('-bilung.nii.gz')
            # print("axial_lesion_filename: ", axial_lesion_filename)

            axial_lesion_scan_input_path = os.path.join(axial_lesion_folder, str(axial_lesion_filename) + '-' + str(slice_position) + '.nii.gz')
            # print('axial_lesion_scan_input_path: ', axial_lesion_scan_input_path)

            ## Reads .nii file and get the numpy image array
            axial_lesion = nib.load(axial_lesion_scan_input_path)
            axial_lesion_array = axial_lesion.get_data()
            axial_lesion_affine = axial_lesion.affine

            # print('+ axial_lesion_array', axial_lesion.header)
            # print('+ axial_lesion_array', axial_lesion.shape)

            axial_lesion_reshape = np.reshape(axial_lesion_array, (512, 512))
            # print(axial_lesion_reshape.shape)

            ## Adding slices
            new_3D_lesion_array[:, :, slice_position] = axial_lesion_reshape


        # print("new_3D_lesion_array: ", new_3D_lesion_array.shape)
        # print('lesion_scan_array.shape[2]: ', lesion_scan_array.shape[2])
        new_3D_lesion_array_reshape = new_3D_lesion_array.reshape((512, 512, lesion_scan_array.shape[2]))
        # print("new_3D_lesion_array_reshape: ", new_3D_lesion_array_reshape.shape)

        ## Generate the 3D lesions images
        new_3D_lesion_nifti = nib.Nifti1Image(new_3D_lesion_array_reshape, lesion.affine)

        ## Set new name
        new_3D_lesion_filename = str(axial_lesion_filename) + '-3Dlesions.nii.gz'
        ct_slice_output_path = os.path.join(labelsTr_shape_path, new_3D_lesion_filename)

        ## and write the nifti files
        nib.save(new_3D_lesion_nifti, ct_slice_output_path)


def build_3D_images_GT(split='train', task='bi-lung'):

    ## Testbed path
    data_testbed_root = "/data/01_UB/01_Multiomics-Data/Clinical_Imaging/02_Step-3_122-CasesSegmented/"

    ## Dataset path definitions
    nifti_folder = os.path.join(data_testbed_root, "01_Nifti-Data/")
    # lesion_folder = os.path.join(data_testbed_root, "02_Nifti-Seg-6-Classes/")
    lesion_folder = os.path.join(data_testbed_root, "03_Nifti-LungSeg/")

    ## Metadat path
    metadata_path = "/home/jagh/Documents/01_UB/10_Conferences_submitted/08_MIA-Explainable_2021/02_materials/dataset/"
    metadata_file_path = os.path.join(metadata_path, "1XX_dataframe_axial_slices-Temp.csv")
    axial_metadata_file_path= os.path.join(metadata_path, "1XX_dataframe_axial_slices-Temp.csv")

    ## Read the metadata
    metadata_full = pd.read_csv(metadata_file_path, sep=',')
    print("+ Metadata: ", metadata_full.shape)

    ## nn-UNet sandbox paths
    dataset_folder = '/data/01_UB/00_Dev/nnUNet_sandbox_bi-lungs/nnUNet_raw_data/Task115_COVIDSegChallenge/'

    ## Axial slices folder
    if split == 'test':
        axial_lesion_folder = os.path.join(dataset_folder, "AAxial-labelsTs/")
        labelsTr_shape_path = os.path.join(dataset_folder, "3D-labelsTs-GT")
        metadata = metadata_full.query('split == "test"')
    else:
        axial_lesion_folder = os.path.join(dataset_folder, "AAxial-labelsTr/")
        labelsTr_shape_path = os.path.join(dataset_folder, "3D-labelsTr-GT")
        metadata = metadata_full.query('split == "train"')

    ## Create a directory for images and labels inside the Sandbox directory
    Utils().mkdir(labelsTr_shape_path)

    ## Using separate folder for training and test
    metadata = metadata.drop_duplicates(subset=['id_case'])
    metadata = metadata.reset_index(drop=True)
    # print("++ metadata:", metadata.head())
    print("+ Metadata {}: {}".format(split, metadata.shape))

    ## Loop through dataset
    for row in range(metadata.shape[0]):
        # print('+ row: ', row)
        # print('+ ct_file_name: ', metadata['ct_file_name'][row])

        ## locating the CT and Seg
        ct_scan_input_path = os.path.join(nifti_folder, metadata['ct_file_name'][row])

        if task == 'bi-lung':
            lesion_scan_input_path = os.path.join(lesion_folder, metadata['dnn_lung_file_name'][row])
        else:
            lesion_scan_input_path = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

        # print('+ ct_scan_input_path', ct_scan_input_path)
        # print('+ lesion_scan_input_path', lesion_scan_input_path)

        ## Reads .nii file and get the numpy image array
        ct = nib.load(ct_scan_input_path)
        ct_scan_array = ct.get_data()
        # print('+ ct_scan_array', ct.header)

        lesion = nib.load(lesion_scan_input_path)
        lesion_scan_array = lesion.get_data()
        # print("lesion_scan_array: ", lesion_scan_array.shape)


        #####################################################################
        #####################################################################
        ## New
        new_3D_lesion_array = np.zeros_like(lesion_scan_array)
        # print("new_3D_lesion_array: ", new_3D_lesion_array.shape)

        for slice in range(lesion_scan_array.shape[2]):

            slice_position = slice
            axial_lesion_filename, _ = str(metadata['lesion_file_name'][row]).split('-bilung.nii.gz')
            # print("axial_lesion_filename: ", axial_lesion_filename)


            ## Set file path of the 2D axial lesion
            axial_lesion_scan_input_path = os.path.join(axial_lesion_folder, str(axial_lesion_filename) + '-' + str(slice_position) + '.nii.gz')
            # print('axial_lesion_scan_input_path: ', axial_lesion_scan_input_path)

            ## Reads .nii file and get the numpy image array
            axial_lesion = nib.load(axial_lesion_scan_input_path)
            axial_lesion_array = axial_lesion.get_data()
            axial_lesion_affine = axial_lesion.affine

            # print('+ axial_lesion_array', axial_lesion.header)
            # print('+ axial_lesion_array', axial_lesion.shape)

            axial_lesion_reshape = np.reshape(axial_lesion_array, (512, 512))

            ##############################################################################
            ##############################################################################
            ## Added GT

            ## Get the GT axial slice positions
            loc_id_case = metadata['id_case'][row]
            # axial_loc = axial_metadata[axial_metadata['id_case'] == 'B0018_01_200410_CT_SK']
            gt_loc = metadata[metadata['id_case'] == loc_id_case]
            # print('axial_loc', axial_loc)

            ## Get the ground truht
            gt_slices = gt_loc['slice_position'].values
            gt_slices = gt_slices -1
            # print('axial_loc', gt_loc['slice_position'].values)
            # print('gt_slices', gt_slices)

            gt_lesions = gt_loc['slice_with_lesion'].values
            # print(gt_lesions)

            if slice in gt_slices:
                # print('slice', slice)
                gt_lesion_array = lesion_scan_array[:, :, slice]

                ## Adding slices
                new_3D_lesion_array[:, :, slice_position] = gt_lesion_array

            else:
                ## Adding slices
                # new_3D_lesion_array[slice_position] = np.append(new_3D_lesion_array, axial_lesion_reshape)
                new_3D_lesion_array[:, :, slice_position] = axial_lesion_reshape


            ## Added GT
            ##############################################################################
            ##############################################################################

        # print("new_3D_lesion_array: ", new_3D_lesion_array.shape)
        # new_3D_lesion_array_reshape = new_3D_lesion_array.reshape((512, 512, lesion_scan_array.shape[2]))
        # print("new_3D_lesion_array_reshape: ", new_3D_lesion_array_reshape.shape)

        new_3D_lesion_array_reshape = new_3D_lesion_array

        ## Generate the 3D lesions images
        new_3D_lesion_nifti = nib.Nifti1Image(new_3D_lesion_array_reshape, lesion.affine)

        ## Set new name
        new_3D_lesion_filename = str(axial_lesion_filename) + '-3Dlesions.nii.gz'
        ct_slice_output_path = os.path.join(labelsTr_shape_path, new_3D_lesion_filename)

        ## and write the nifti files
        nib.save(new_3D_lesion_nifti, ct_slice_output_path)



def run(args):
    """
    Run the pipeline to build a 2D dataset from CT scans
    :param dataframe: Path to the dataframe containing the CT scans
    :param cts_folder: Path to the folder containing the CT scans
    :param seg_folder: Path to the folder containing the segmentations
    :param sandbox: Path to the sandbox where intermediate results are stored
    """
    # convert_axial_slices(args.dataframe, args.cts_folder, args.sandbox, "train")
    # convert_axial_slices(args.dataframe, args.cts_folder, args.sandbox, "test")

    # build_3D_images('train')
    # build_3D_images('test')

    build_3D_images_GT('train')
    # build_3D_images_GT('test')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge')
    parser.add_argument('-d', '--dataframe', default='/home/jagh/Documents/01_conferences_submitted/00_MIA-Explainable_2021/dataset/122_dataframe_axial_slices.csv')
    parser.add_argument('-cts', '--cts_folder', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/02_Step-3_122-CasesSegmented/01_Nifti-Data/')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
