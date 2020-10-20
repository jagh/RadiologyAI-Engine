import glob
import collections
import os, pickle, math
import dicom2nifti


class Utils:
    """
    Module for defining utility functions
    """

    def __init__(self):
        pass


    def mkdir(self, directory):
         if not os.path.exists(directory):
             os.makedirs(directory)


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

    def read_nii(self, filepath):
        '''
        Reads .nii file and returns pixel array
        '''
        ct_scan = nib.load(filepath)
        array   = ct_scan.get_fdata()
        array   = np.rot90(np.array(array))
        return (array)


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
