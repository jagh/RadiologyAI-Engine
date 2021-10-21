
import nibabel as nib
import numpy as np


class ImagePreprocessing:
    """
    Module with preprocesing transformation of CTs
    """

    def __init__(self):
        pass

    def relabel_sequential(self, lesion_nifti_file, sequence = [0, 1, 2, 4, 5, 6]):
        """
        Relabel the lesion segmentation using the given sequence.
        Parameters
        ----------
        lesion_nifti_file : str
            Path to the lesion segmentation file.
        sequence : list
            The sequence of the labels.

        Returns
        -------
        relabel_lesion_nifti : Nifti Image
            The relabeled lesion segmentation.
        """

        ## 3D lesion scan load
        lesion = nib.load(lesion_nifti_file)
        lesion_array = lesion.get_fdata()
        lesion_affine = lesion.affine
        # print("++ lesion_affine:", lesion_affine.shape)

        try:
            ## new lesion array
            relabel_lesion = np.zeros_like(lesion_array)

            ## Change the sequence
            new_sequence = 0
            for seq in sequence:
                relabel_lesion[lesion_array == seq] = new_sequence
                new_sequence += 1


            relabel_lesion_nifti = nib.Nifti1Image(relabel_lesion, lesion_affine)
            return relabel_lesion_nifti

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")
