
import sys, os
import six

# import itk
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

    def read_nrrd_file(self, file_path):
        reader = sitk.ImageFileReader()
        reader.SetImageIO('NrrdImageIO')
        reader.SetFileName(file_path)
        return reader.Execute();


    def feature_extractor(self, ct_image_path, ct_mask_path, study_name, label_name):
        """
        Using pyradiomics to extract first order, shape, texture and other features
        + Reference work: Griethuysen et al. Computational Radiomics System to Decode theRadiographic Phenotype. 2017.
            -> https://cancerres.aacrjournals.org/content/canres/77/21/e104.full.pdf
        """

        ## nifti images
        ct_image = sitk.ReadImage(ct_image_path)
        ct_mask = sitk.ReadImage(ct_mask_path)

        # # ## nrrd sources
        # ct_image = self.read_nrrd_file(ct_image_path)
        # ct_mask = self.read_nrrd_file(ct_mask_path)

        image_feature_list = []
        image_feature_list.append(study_name)
        image_feature_list.append(label_name)

        image_header_list = []
        image_header_list.append("study_name")
        image_header_list.append("label_name")

        ## Get the First Order features
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(ct_image, ct_mask, **self.settings)
        extractor_firstOrder = firstOrderFeatures.execute()
        for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("fOF_" + key))
            image_header_list.append(str("AA_" + key))


        ## Get Shape Features in 3D
        shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask, **self.settings)
        extractor_shape = shapeFeatures3D.execute()
        for (key, val) in six.iteritems(shapeFeatures3D.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("sF3D_" + key))
            image_header_list.append(str("BB_" + key))


        glcmFeatures = glcm.RadiomicsGLCM(ct_image, ct_mask, **self.settings)
        extractor_glcm = glcmFeatures.execute()
        for (key, val) in six.iteritems(glcmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("glcmF_" + key))
            image_header_list.append(str("CC_" + key))


        # glszmFeatures = glszm.RadiomicsGLSZM(ct_image, ct_mask, **self.settings)
        # extractor_glszm = glszmFeatures.execute()
        # for (key, val) in six.iteritems(glszmFeatures.featureValues):
        #     # print("\t%s: %s" % (key, val))
        #     image_feature_list.append(val)


        glrlmFeatures = glrlm.RadiomicsGLRLM(ct_image, ct_mask, **self.settings)
        extractor_glrlm = glrlmFeatures.execute()
        for (key, val) in six.iteritems(glrlmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("glcmF_" + key))
            image_header_list.append(str("DD_" + key))


        ngtdmFeatures = ngtdm.RadiomicsNGTDM(ct_image, ct_mask, **self.settings)
        extractor_ngtdm = ngtdmFeatures.execute()
        for (key, val) in six.iteritems(ngtdmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("ngtdmF_" + key))
            image_header_list.append(str("EE_" + key))


        gldmFeatures = gldm.RadiomicsGLDM(ct_image, ct_mask, **self.settings)
        extractor_gldm = gldmFeatures.execute()
        for (key, val) in six.iteritems(gldmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("ngtdmF_" + key))
            image_header_list.append(str("FF_" + key))


        return image_feature_list, image_header_list
