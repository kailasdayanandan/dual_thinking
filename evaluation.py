###################################################################
##
##              Matching and calculation
##
###################################################################

import numpy as np
from utils import *


def get_matches(mask, segms):
    match_indexes, match_scores = search_segments(mask, segms)
    if len(match_indexes) > 1:            
        match_scores, match_indexes = zip(*sorted(zip(match_scores, match_indexes), reverse=True))                    
    return match_indexes, match_scores

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def search_score(mask, segments):
    max_iou, max_idx = 0,0
    for i, segment in enumerate(segments):
        segment_b = segment.astype(bool)
        iou_score = iou(segment_b, mask)
        if iou_score > max_iou:
            max_iou = iou_score
            max_idx = i
    return max_iou, max_idx   

def search_segments(mask, segments):
    match_indexes = []
    match_scores = []

    for i, segment in enumerate(segments):
        segment_b = segment.astype(bool)
        iou_score = iou(segment, mask)
        if iou_score > 0.1:
            match_indexes.append(i)
            match_scores.append(iou_score)

    return match_indexes, match_scores

############################### EVALUATION CODE ##############################

def evaluate_first_not_found_impl(img, segms, mask_r1, mask_r2, confusion, found_thresh = 0.2):
    match_indexes, match_scores = get_matches(mask_r1, segms)
    result = True
    if len(match_indexes) == 0:
        return result
    if (match_scores[0] < found_thresh):
        return result    
    return False



def evaluate_first_impl(img, segms, mask_r1, mask_r2, confusion, match_thresh = 0.73):
    match_indexes, match_scores = get_matches(mask_r1, segms)
    result = False
    if len(match_indexes) == 0:
        return result
    if (match_scores[0] >= match_thresh):
        result = True    
    return result

def evaluate_first_second_impl(img, segms, mask_r1, mask_r2, confusion, match_thresh = 0.73):
    _, match_scores = get_matches(mask_r1, segms)
    _, match_scores2 = get_matches(mask_r2, segms)
    result = False
    if len(match_scores) > 0 and  (match_scores[0] >= match_thresh) and len(match_scores2) > 0  and (match_scores2[0] >= match_thresh):
        result = True
    return result

def evaluate_confusion_in_first_impl(img, segms, mask_r1, mask_r2, confusion, overlap_thresh=0.70, match_thresh = 0.73):

    match_indexes, match_scores = get_matches(mask_r1, segms)    
    rslt = False

    if len(match_indexes) == 0:
        return rslt
    if (match_scores[0] < match_thresh):
        return rslt

    overlap = segms[match_indexes[0]] * confusion
    score = iou(overlap,confusion)   
    if score > overlap_thresh:
        return True

    return rslt

def evaluate_confusion_not_in_first_impl(img, segms, mask_r1, mask_r2, confusion, overlap_thresh=0.25, match_thresh = 0.73):

    maskr1_wc = np.logical_and(mask_r1.astype(bool), np.logical_not(confusion.astype(bool))) * 255
    match_indexes, match_scores = get_matches(maskr1_wc, segms)    
    rslt = False

    if len(match_indexes) == 0:
        return False
    if (match_scores[0] < match_thresh):
        return rslt
    
    overlap = segms[match_indexes[0]] * confusion
    score = iou(overlap,confusion)    
    if score < overlap_thresh:
        rslt = True

    return rslt

def evaluate_human_and_not_confusion_impl(img, segms, mask_r1, mask_r2, confusion, mask_h, overlap_thresh=0.25, match_thresh = 0.73):

    mask_h_wc = np.logical_and(mask_h.astype(bool), np.logical_not(confusion.astype(bool))) * 255
    match_indexes, match_scores = get_matches(mask_h_wc, segms)    

    if len(match_indexes) == 0:
        return False    
    if (match_scores[0] < match_thresh):
        return False
    
    overlap = segms[match_indexes[0]] * confusion
    score = iou(overlap,confusion)    
    if score < overlap_thresh:        
        return True

    return False


def evaluate_human_and_confusion_impl(img, segms, mask_r1, mask_r2, confusion, mask_h, overlap_thresh=0.7, match_thresh = 0.73):

    match_indexes, match_scores = get_matches(mask_h, segms)    
    rslt = False

    if len(match_indexes) == 0:
        return rslt    
    if (match_scores[0] < match_thresh):
        return rslt    

    overlap = segms[match_indexes[0]] * confusion
    score = iou(overlap,confusion)    
    if score < overlap_thresh:        
        return rslt

    return True


def evaluate_human_first_second_impl(img, segms, mask_r1, mask_r2, mask_h, overlap_thresh=0.7, match_thresh = 0.73):
    
    rslt = False

    match_indexes, match_scores = get_matches(mask_h, segms)
    if len(match_indexes) == 0:
        return rslt    
    if (match_scores[0] < match_thresh):
        return rslt

    overlap = segms[match_indexes[0]] * mask_r1
    score = iou(overlap,mask_r1)    
    if score < overlap_thresh:        
        return rslt

    overlap = segms[match_indexes[0]] * mask_r2
    score = iou(overlap,mask_r2)    
    if score < overlap_thresh:   
        return rslt

    return True

