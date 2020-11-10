
import sys, os
import six

import SimpleITK as sitk
from radiomics import featureextractor
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm

class RadiomicsExtractor:
    """ Extraction of radiomics feactures from a CT images """

    def __init__(self, lobes_area):
        self.settings = {}
        self.settings['binWidth'] = 25
        self.settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        self.settings['interpolator'] = sitk.sitkBSpline
        self.settings['label'] = int(lobes_area)


    def feature_extractor(self, ct_image_path, ct_mask_path, study_name, label_name):
        """
        Using pyradiomics to extract first order and shape features
        """

        ct_image = sitk.ReadImage(ct_image_path)
        ct_mask = sitk.ReadImage(ct_mask_path)
        image_feature_list = []

        image_feature_list.append(study_name)
        image_feature_list.append(label_name)


        ## Get the First Order features
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(ct_image, ct_mask, **self.settings)
        extractor_firstOrder = firstOrderFeatures.execute()
        for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        ## Get Shape Features in 3D
        shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask, **self.settings)
        extractor_shape = shapeFeatures3D.execute()
        for (key, val) in six.iteritems(shapeFeatures3D.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        glcmFeatures = glcm.RadiomicsGLCM(ct_image, ct_mask, **self.settings)
        extractor_glcm = glcmFeatures.execute()
        for (key, val) in six.iteritems(glcmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        glszmFeatures = glszm.RadiomicsGLSZM(ct_image, ct_mask, **self.settings)
        extractor_glszm = glszmFeatures.execute()
        for (key, val) in six.iteritems(glszmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        glrlmFeatures = glrlm.RadiomicsGLRLM(ct_image, ct_mask, **self.settings)
        extractor_glrlm = glrlmFeatures.execute()
        for (key, val) in six.iteritems(glrlmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        ngtdmFeatures = ngtdm.RadiomicsNGTDM(ct_image, ct_mask, **self.settings)
        extractor_ngtdm = ngtdmFeatures.execute()
        for (key, val) in six.iteritems(ngtdmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        gldmFeatures = gldm.RadiomicsGLDM(ct_image, ct_mask, **self.settings)
        extractor_gldm = gldmFeatures.execute()
        for (key, val) in six.iteritems(gldmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)


        return image_feature_list
