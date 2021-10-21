
import nibabel as nib
import numpy as np


class ImagePreprocessing:
    """
    Module with preprocesing transformation of CTs
    """

    def __init__(self):
        pass

    def relabel_segmentation(self, lesion_nifti_file):
        """ Method to relabel each lesion segmentation"""

        ## 3D lesion scan load
        lesion = nib.load(lesion_nifti_file)
        lesion_array = lesion.get_fdata()
        lesion_affine = lesion.affine
        # print("++ lesion_affine:", lesion_affine.shape)

        try:
        ## Add relabel_lesion_folder
            relabel_lesion = np.zeros_like(lesion_array)
            relabel_lesion[lesion_array == 0] = 0
            relabel_lesion[lesion_array == 1] = 1
            relabel_lesion[lesion_array == 2] = 2
            # relabel_lesion[lesion_array == 3] = 3
            relabel_lesion[lesion_array == 4] = 3
            relabel_lesion[lesion_array == 5] = 4
            relabel_lesion[lesion_array == 6] = 5
            # relabel_lesion[lesion_array == 7] = 6

            relabel_lesion_nifti = nib.Nifti1Image(relabel_lesion, lesion_affine)
            return relabel_lesion_nifti

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")
