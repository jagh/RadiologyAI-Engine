
import sys, os
import six

import SimpleITK as sitk
from radiomics import featureextractor
from radiomics import firstorder, shape

class RadiomicsExtractor:
    """ Extraction of radiomics feactures from a CT images """

    def __init__(self):
        pass


    def feature_extractor(self, ct_image_path, ct_mask_path, study_name, label):
        """
        Using pyradiomics to extract first order and shape features
        """
        ct_image = sitk.ReadImage(ct_image_path)
        ct_mask = sitk.ReadImage(ct_mask_path)
        image_feature_list = []

        image_feature_list.append(study_name)
        image_feature_list.append(label)


        ## Get the First Order features
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(ct_image, ct_mask)
        extractor_firstOrder = firstOrderFeatures.execute()
        for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        ## Get Shape Features in 3D
        shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask)
        extractor_shape = shapeFeatures3D.execute()
        for (key, val) in six.iteritems(shapeFeatures3D.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)

        return image_feature_list
