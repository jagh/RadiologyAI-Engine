
import sys, os

import SimpleITK as sitk
from engine.utils import Utils
from engine.third_party.lungmask import mask
from engine.third_party.lungmask import resunet


class LungSegmentations:
    """
    Lung lobes segmentation using a lungmask module
    """

    def __init__(self):
        print("++ Welcome to Lung Segmentation")
        pass

    def file_segmentation(self, input_ct, seg_method='lobes', batch_size=10):
        """
        CT lung lobes segmentation using UNet
        """

        if seg_method == 'bi-lung':
            model = mask.get_model('unet', 'R231')
            print('R231-method')
        elif seg_method == 'lobes':
            model = mask.get_model('unet', 'LTRCLobes')
        else:
            print('Segmentation method not found!')

        ct_segmentation = mask.apply(input_ct, model, batch_size=batch_size)

        ### Write segmentation
        result_out = sitk.GetImageFromArray(ct_segmentation)
        result_out.CopyInformation(input_ct)
        # result_out = np.rot90(np.array(result_out)) ## modifyed the orientation

        return result_out



    def folder_segmentations(self, input_folder, output_folder, seg_method='lobes', batch_size=10):
        """
        CT Lung lobes segmentation for all nii.gz files within the directory.
        """
        for input_path in input_folder:

            # print("engine - input_path", input_path)

            ct_name = input_path.split(os.path.sep)[-1]
            ct_dcm_format = str(ct_name.split('.nii.gz')[0] + '-' + seg_method + '.nii.gz')

            input_ct = sitk.ReadImage(input_path)
            result_out = self.file_segmentation(input_ct, seg_method, batch_size)

            Utils().mkdir(output_folder)
            sitk.WriteImage(result_out, str(output_folder+"/"+ct_dcm_format))
            print("CT segmentation file: {}".format(str(output_folder+"/"+ct_dcm_format)))
