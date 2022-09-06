import sys, os
from medpy import metric

class MIM:
    """ Processing of a variety of medical imaging metrics based on the MedPy library 
        -----------------------
        Parameters:
        - result: the result of the segmentation
        - reference: the ground truth
        - patient_id: the patient id
        - slice_position: the slice position
    """

    def __init__(self):
        self.mim_values_list = []
        self.mim_header_list = []

    def binary_metrics(self, result, reference, patient_id, slice_position, who_score, seg_class):

        self.mim_values_list.append(patient_id)
        self.mim_values_list.append(slice_position)
        self.mim_values_list.append(who_score)
        self.mim_values_list.append(seg_class)

        self.mim_header_list.append('patient_id')
        self.mim_header_list.append('slice_position')
        self.mim_header_list.append('who_score')
        self.mim_header_list.append('seg_class')

        dice_coefficient = metric.binary.dc(result, reference)
        self.mim_values_list.append(dice_coefficient)
        self.mim_header_list.append('dice_coefficient')

        try:
            jaccard_coefficient = metric.binary.jc(result, reference)
        except ZeroDivisionError:
            jaccard_coefficient = 0.0
        self.mim_values_list.append(jaccard_coefficient)
        self.mim_header_list.append('jaccard_coefficient')

        try:
            hausdorff_distance = metric.binary.hd(result, reference)
        except RuntimeError:
            hausdorff_distance = 0.0
        self.mim_values_list.append(hausdorff_distance)
        self.mim_header_list.append('hausdorff_distance')

        try:
            average_surface_distance = metric.binary.asd(result, reference)
        except RuntimeError:
            average_surface_distance = 0.0
        self.mim_values_list.append(average_surface_distance)
        self.mim_header_list.append('avg_surface_distance')

        try:
            average_symmetric_surface_distance = metric.binary.assd(result, reference)
        except RuntimeError:
            average_symmetric_surface_distance = 0.0
        self.mim_values_list.append(average_symmetric_surface_distance)
        self.mim_header_list.append('avg_symmetric_surface_distance')

        sensitivity = metric.binary.sensitivity(result, reference)
        self.mim_values_list.append(sensitivity)
        self.mim_header_list.append('sensitivity')

        specificity = metric.binary.specificity(result, reference)
        self.mim_values_list.append(specificity)
        self.mim_header_list.append('specificity')

        try:
            relative_absolute_volume_difference = metric.binary.ravd(result, reference)
        except RuntimeError:
            relative_absolute_volume_difference = 0.0
        self.mim_values_list.append(relative_absolute_volume_difference)
        self.mim_header_list.append('rel_absolute_volume_difference')

        return self.mim_values_list, self.mim_header_list
