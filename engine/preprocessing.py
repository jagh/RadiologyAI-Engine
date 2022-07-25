
import nibabel as nib
import numpy as np

import SimpleITK as sitk


class SegProcessing:
    """
    Module for preprocesing the CT segmentations
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

    def label_overlap(self, lesion_nifti_file, sequenceOn = [1, 2, 3, 4], sequenceOff = [0, 5]):
        """
        This function takes a 3D lesion segmentation file and change the label values according to the sequence.
        Parameters
        ----------
        lesion_nifti_file : str
            Path to the lesion segmentation file.
        sequence : list(int)
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
            new_label = np.zeros_like(lesion_array)

            ## Change the  sequence on
            for seq in sequenceOn:
                new_label[lesion_array == seq] = 1

            ## Remove the sequence off
            for seq in sequenceOff:
                new_label[lesion_array == seq] = 0


            new_label_nifti = nib.Nifti1Image(new_label, lesion_affine)
            return new_label_nifti

        except(Exception, ValueError) as e:
            print("Not lesion segmentation")




class ImageProcessing:
        """
        Module for preprocesing the CT images
        """

        def __init__(self):
            pass

        def extract_axial_slice_3D(self, nifti_file_name, axial_index):
            """
            This function extracts the axial slice from a 3D nifti file.
            The function takes in the nifti file name and the axial index
            and returns the axial slice.

            Parameters
            ----------
            nifti_file_name : str
                The nifti file name.
            axial_index : int
                The axial index of the slice.
            Returns
            -------
            image_slice : nifti image
                The axial slice of the nifti file.
            """

            ## get the ct array
            image = nib.load(nifti_file_name)
            image_array = image.get_fdata()
            image_affine = image.affine

            ## Get the axial slice in array for images and labels
            image_slice = image_array[:, :, axial_index]

            # ## Axial slice transformation with shape (x, y, 1) -> 'patch_size':([  1, 512])
            # image_array_reshape = image_slice.reshape((512, 512, 1))

            ## Axial slice transformation with shape (1, x, y) ->
            image_array_reshape = image_slice.reshape((1, 512, 512))

            return nib.Nifti1Image(image_array_reshape, image_affine)

        def extract_3D_slices(self, nifti_file_name, axial_index):
            """
            This function extracts the axial slice from a 3D nifti file.
            The function takes in the nifti file name and the axial index
            and returns the axial slice.

            Parameters
            ----------
            nifti_file_name : str
                The nifti file name.
            axial_index : int
                The axial index of the slice.
            Returns
            -------
            image_slice : nifti image
                The axial slice of the nifti file.
            """

            ## SimpleITK -> get the ct array
            image_array = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file_name))
            image_itk = sitk.ReadImage(nifti_file_name)

            try:
                ## Get the axial slice in array for images and labels
                image_slice = image_array[axial_index, :, :]

                ## SimpleITK for axial slice transformation with shape (1, x, y)
                image_array_reshape = image_slice.reshape((1, 512, 512))

                # axial_slice_3D = sitk.GetImageFromArray(image_array_reshape)
                # axial_slice_3D.CopyInformation(seg_itk)

                return sitk.GetImageFromArray(image_array_reshape)

            except IndexError:
                print("+ IndexError: index {} is out of bounds in {}".format(axial_index, nifti_file_name))
