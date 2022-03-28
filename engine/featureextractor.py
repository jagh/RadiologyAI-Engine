
import sys, os
import six
import pandas as pd

# import itk
import SimpleITK as sitk
import nibabel as nib


from radiomics import featureextractor
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm, shape2D

class RadiomicsExtractor:
    """ Extraction of radiomics feactures from a CT images """

    def __init__(self, lobes_area):
        self.settings = {}
        self.settings['binWidth'] = 20
        self.settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        self.settings['interpolator'] = sitk.sitkBSpline
        self.settings['label'] = int(1)

    def read_nrrd_file(self, file_path):
        reader = sitk.ImageFileReader()
        reader.SetImageIO('NrrdImageIO')
        reader.SetFileName(file_path)
        return reader.Execute();

    def feature_extractor_2D(self, ct_image_path, ct_mask_path, study_name, label_name):
        """
        Using pyradiomics to extract first order, shape, texture and other features
        + Reference work: Griethuysen et al. Computational Radiomics System to Decode theRadiographic Phenotype. 2017.
            -> https://cancerres.aacrjournals.org/content/canres/77/21/e104.full.pdf
        """

        # ## nifti images
        ct_image = sitk.ReadImage(ct_image_path)
        ct_mask = sitk.ReadImage(ct_mask_path)

        # nib.Nifti1Image(output, func.affine)

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


        ## Get Shape Features in 2D
        shapeFeatures2D = shape2D.RadiomicsShape2D(ct_image, ct_mask, **self.settings)
        extractor_shape = shapeFeatures2D.execute()
        for (key, val) in six.iteritems(shapeFeatures2D.featureValues):
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


        glszmFeatures = glszm.RadiomicsGLSZM(ct_image, ct_mask, **self.settings)
        extractor_glszm = glszmFeatures.execute()
        for (key, val) in six.iteritems(glszmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)
            image_header_list.append(str("GG_" + key))


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

    def feature_extractor_3D(self, ct_image_path, ct_mask_path, study_name, label_name):
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


        glszmFeatures = glszm.RadiomicsGLSZM(ct_image, ct_mask, **self.settings)
        extractor_glszm = glszmFeatures.execute()
        for (key, val) in six.iteritems(glszmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)
            image_header_list.append(str("GG_" + key))


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



########################################################################################
########################################################################################
    ## aa_
    def get_firstOrderFeatures(self, ct_image, ct_mask, image_feature_list, image_header_list):
        """ Get the First Order features """
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(ct_image, ct_mask, **self.settings)
        extractor_firstOrder = firstOrderFeatures.execute()
        for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            image_header_list.append(str("aa_" + key))
        return image_feature_list, image_header_list

    # ## bb_
    # def get_shapeFeatures3DBK(self, ct_image, ct_mask, image_feature_list, image_header_list):
    #     """ Get Shape Features in 3D """
    #     shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask, **self.settings)
    #     extractor_shape = shapeFeatures3D.execute()
    #     for (key, val) in six.iteritems(shapeFeatures3D.featureValues):
    #         # print("key: {} || val: {}".format(key, val))
    #         image_feature_list.append(val)
    #         image_header_list.append(str("bb_" + key))
    #     return image_feature_list, image_header_list
    #
    # ## bb_
    # def get_shapeFeatures3D_CoreDumped(self, ct_image, ct_mask, image_feature_list, image_header_list):
    #     """ Get Shape Features in 3D """
    #     shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask, **self.settings)
    #     # shapeFeatures3D.enableAllFeatures()
    #     # shapeFeatures3D.disableAllFeatures()
    #     shapeFeatures3D.Volume()
    #     shapeFeatures3D.execute()
    #     for (key, val) in six.iteritems(shapeFeatures3D.featureValues):
    #         # print("key: {} || val: {}".format(key, val))
    #         image_feature_list.append(val)
    #         image_header_list.append(str("bb_" + key))
    #     return image_feature_list, image_header_list


    ## bb_
    def get_shapeFeatures3D(self, ct_image_path, ct_mask_path, bb_filename_case):
        """ Get Shape Features in 3D """

        print("////////////////////////////////////////////////////")
        print("+ bb_filename_case: ", bb_filename_case)

        ## nifti images
        ct_image = sitk.ReadImage(ct_image_path)
        ct_mask = sitk.ReadImage(ct_mask_path)

        # shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask, **self.settings)
        shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask)
        shapeFeatures3D.enableAllFeatures()
        shapeFeatures3D.execute()
        df_shape = pd.DataFrame(shapeFeatures3D.featureValues, index=[0])
        df_shape = df_shape.add_prefix('bb_')

        print("+ df_shape: ", df_shape)
        print("+ Type(df_shape) : ", type(df_shape))

        df_shape.to_csv(bb_filename_case, sep=',', index=False)
        return df_shape

    def get_shapeFeatures2D(self, ct_image, ct_mask, image_feature_list, image_header_list):
        ## Get Shape Features in 3D
        shapeFeatures2D = shape.RadiomicsShape(ct_image, ct_mask, **self.settings)
        extractor_shape = shapeFeatures2D.execute()
        for (key, val) in six.iteritems(shapeFeatures2D.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("sF3D_" + key))
            image_header_list.append(str("bb_" + key))
        return image_feature_list, image_header_list

    ## cc_
    def get_glcmFeatures(self, ct_image, ct_mask, image_feature_list, image_header_list):
        glcmFeatures = glcm.RadiomicsGLCM(ct_image, ct_mask)    #, **self.settings)
        extractor_glcm = glcmFeatures.execute()
        for (key, val) in six.iteritems(glcmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("glcmF_" + key))
            image_header_list.append(str("cc_" + key))
        return image_feature_list, image_header_list

    ## gg_
    def get_glszmFeatures(self, ct_image, ct_mask, image_feature_list, image_header_list):
        glszmFeatures = glszm.RadiomicsGLSZM(ct_image, ct_mask, **self.settings)
        extractor_glszm = glszmFeatures.execute()
        for (key, val) in six.iteritems(glszmFeatures.featureValues):
            # print("\t%s: %s" % (key, val))
            image_feature_list.append(val)
            image_header_list.append(str("gg_" + key))
        return image_feature_list, image_header_list

    ## dd_
    def get_glrlmFeatures(self, ct_image, ct_mask, image_feature_list, image_header_list):
        glrlmFeatures = glrlm.RadiomicsGLRLM(ct_image, ct_mask, **self.settings)
        extractor_glrlm = glrlmFeatures.execute()
        for (key, val) in six.iteritems(glrlmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("glcmF_" + key))
            image_header_list.append(str("dd_" + key))
        return image_feature_list, image_header_list

    ## ee_
    def get_ngtdmFeatures(self, ct_image, ct_mask, image_feature_list, image_header_list):
        ngtdmFeatures = ngtdm.RadiomicsNGTDM(ct_image, ct_mask, **self.settings)
        extractor_ngtdm = ngtdmFeatures.execute()
        for (key, val) in six.iteritems(ngtdmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("ngtdmF_" + key))
            image_header_list.append(str("ee_" + key))
        return image_feature_list, image_header_list

    ## ff_
    def get_gldmFeatures(self, ct_image, ct_mask, image_feature_list, image_header_list):
        gldmFeatures = gldm.RadiomicsGLDM(ct_image, ct_mask, **self.settings)
        extractor_gldm = gldmFeatures.execute()
        for (key, val) in six.iteritems(gldmFeatures.featureValues):
            # print("key: {} || val: {}".format(key, val))
            image_feature_list.append(val)
            # image_header_list.append(str("ngtdmF_" + key))
            image_header_list.append(str("ff_" + key))
        return image_feature_list, image_header_list


    ########
    def parallel_extractor(self, ct_image_path, ct_mask_path,
                                study_name, label_name, radiomics_set, bb_filename_case="0"):
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

        # df_shape_3D = pd.DataFrame(shapeFeatures3D.featureValues, index=[0])

        if "aa" == radiomics_set:
            print("+ get_firstOrderFeatures:", radiomics_set)
            image_feature_list, image_header_list = self.get_firstOrderFeatures(ct_image, ct_mask,
                                                image_feature_list, image_header_list)
        elif "3D_bb" == radiomics_set:
            print("!! Get shapeFeatures3D:", radiomics_set)
            df_shape = self.get_shapeFeatures3D(ct_image_path, ct_mask_path, bb_filename_case)
        # elif "bb" == radiomics_set:
        #     print("++ get_shapeFeatures3D:", radiomics_set)
        #     df_shape = self.get_shapeFeatures3D(ct_image, ct_mask, bb_filename_case)

        elif "bb" == radiomics_set:
            print("+ get_shapeFeatures2D:", radiomics_set)
            image_feature_list, image_header_list = self.get_shapeFeatures2D(ct_image, ct_mask,
                                                    image_feature_list, image_header_list)
        elif "cc" == radiomics_set:
            print("+ get_glcmFeatures:", radiomics_set)
            image_feature_list, image_header_list = self.get_glcmFeatures(ct_image, ct_mask,
                                                image_feature_list, image_header_list)
        elif "gg" == radiomics_set:
            print("+ get_glszmFeatures:", radiomics_set)
            image_feature_list, image_header_list = self.get_glszmFeatures(ct_image, ct_mask,
                                                image_feature_list, image_header_list)
        elif "dd" == radiomics_set:
            print("+ get_glrlmFeatures:", radiomics_set)
            image_feature_list, image_header_list = self.get_glrlmFeatures(ct_image, ct_mask,
                                                image_feature_list, image_header_list)
        elif "ee" == radiomics_set:
            print("+ get_ngtdmFeatures", radiomics_set)
            image_feature_list, image_header_list = self.get_ngtdmFeatures(ct_image, ct_mask,
                                                image_feature_list, image_header_list)
        elif "ff" == radiomics_set:
            print("+ get_gldmFeatures:", radiomics_set)
            image_feature_list, image_header_list = self.get_gldmFeatures(ct_image, ct_mask,
                                                image_feature_list, image_header_list)
        else:
            print("+ radiomic set not found!!!")


        ###############################
        if "3D-bb" == radiomics_set:
            return df_shape
        else:
            return image_feature_list, image_header_list
