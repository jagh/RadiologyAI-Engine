import glob
import collections
import os, pickle, math
import dicom2nifti
import SimpleITK as sitk

import nibabel as nib
import numpy as np
import json


class Utils:
    """
    Module for defining utility functions
    """

    def __init__(self):
        pass


    def mkdir(self, directory):
         if not os.path.exists(directory):
             os.makedirs(directory)

    def read_dicom(self, filepath):
        ''' Reads .dicom file and returns pixel array '''
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(filepath)
        reader.SetFileNames(dicom_names)
        return reader.Execute()

    def read_nifti(self, filepath):
        ''' Reads .nii file and returns pixel array '''
        ct_scan = nib.load(filepath)
        array   = ct_scan.get_fdata()
        array   = np.rot90(np.array(array))
        return (array)

    def read_nrrd(self, file_path):
        reader = sitk.ImageFileReader()
        reader.SetImageIO('NrrdImageIO')
        reader.SetFileName(file_path)
        return reader.Execute();

    def write_nifti(self, image, file_name):
        """
        Function to write sitk images
            + Slicer utils: https://github.com/Slicer/Slicer/blob/master/Base/Python/tests/test_sitkUtils.py
        """
        write = sitk.ImageFileWriter()
        write.SetFileName(file_name)
        write.SetImageIO('NiftiImageIO')
        write.Execute(image)

    def get_file_identifiers(self, folder, suffix = '_0000.nii.gz'):
        """
        This function returns a list of file identifiers in a folder.
        """
        files = glob.glob(folder + "/*")
        identifiers = []
        for f in files:
            f, _ = f.split(suffix)
            f = f.split('/')
            identifiers.append(f[-1])

        return identifiers

    def save_json(self, obj, file, indent= 4, sort_keys = True):
        with open(file, 'w') as f:
            json.dump(obj, f, sort_keys=sort_keys, indent=indent)


    def convert_dcm2nii(self, dcm_folder, output_folder):
        """
        Function to convert dicom files to nifti file
        """
        self.inputs = dcm_folder
        self.outputs = output_folder

        for folder_path in dcm_folder:
            ## Get folder name
            folder_name = folder_path.split(os.path.sep)[-1]
            # print(" - folder_name: {}".format(folder_name))

            ## Create the output folder by case
            self.mkdir(str(output_folder + folder_name ))
            nii_folder_path =  str(output_folder + "/" + folder_name + "/")
            # print(" - outpu_folder: {}".format(nii_folder_path))

            #######################################################################
            ## Converting dicom files to nifti
            #######################################################################
            dicom2nifti.convert_directory(str(folder_path + "/"), nii_folder_path)
                                                    #,reorient_nifti=True)

            #######################################################################
            ### Nii file renamed
            #######################################################################
            nii_file_name = glob.glob(str(nii_folder_path + "/*"))
            nii_new_name = str(nii_folder_path + "/" + folder_name + ".nii.gz" )
            os.rename(nii_file_name[0], nii_new_name)
            # print(" ++ nii_file_name: {}".format(nii_file_name[0]))
            # print(" ++ nii_new_name: {}".format(nii_new_name))

    def itk_convert_dcm2nii(self, dcm_folder, output_folder):
        """
        Function to convert dicom files to nifti file
        """
        self.inputs = dcm_folder
        self.outputs = output_folder

        for folder_path in dcm_folder:
            ## Get folder name
            folder_name = folder_path.split(os.path.sep)[-1]
            # print(" - folder_name: {}".format(folder_name))

            ## Create the output folder by case
            self.mkdir(str(output_folder + folder_name ))
            nii_folder_path =  str(output_folder + "/" + folder_name + "/")
            # print(" - output_folder: {}".format(nii_folder_path))

            #######################################################################
            ## Converting dicom files to nifti
            #######################################################################

            ## Read Dicom Image
            # print('++ dicom_folder_path', folder_path)
            dicom_img = self.read_dicom(folder_path)

            ## Write Dicom Image
            nii_file_name = os.path.join(nii_folder_path, str(folder_name + ".nii.gz"))
            self.write_nifti(dicom_img, nii_file_name)
            print('++ nii_file_name', nii_file_name)




#######################################################################
## Launcher settings
#######################################################################

# from utils import Utils
#
# testbed = "testbed/"
# dcm_folder = glob.glob(str(testbed + "/dataset_unibe/sources/*"))
# output_folder = str(testbed + "/dataset_unibe/train-nii/")
#
# Utils().convert_dcm2nii(dcm_folder, output_folder)
