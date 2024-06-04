
import logging
import torch

import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils import *
from evaluation import *

###### EVALUATION CODE #####

def get_operations_mapping():
    operations_d  = {
        'FIRST IDENTIFY'  : evaluate_first,
        'FIRST AND SECOND' : evaluate_first_second,
        'CONFUSION IN FIRST' : evaluate_confusion_in_first,
        'CONFUSION NOT IN FIRST'  : evaluate_confusion_not_in_first,
        'HUMAN IDENTIFY'  : evaluate_human,
        'HUMAN AND CONFUSION' : evaluate_human_and_confusion,
        'HUMAN AND NOT CONFUSION' : evaluate_human_and_not_confusion,
        'HUMAN AND FIRST AND SECOND'  : evaluate_human_first_second,
        'FIRST NOT FOUND'  : evaluate_first_not_found    }
    return operations_d

def evaluate_operation(file, mode, operation, segms, img, display=False,plt_title='', labels=[], c_labels='', mask_dir='./data/annotation/images/'):
    operations_d = get_operations_mapping()
    mask_r1 = load_mask_r1(file, mask_dir=mask_dir)
    mask_r2 = load_mask_r2(file, mask_dir=mask_dir)
    mask_c = load_mask_c(file, mask_dir=mask_dir)
    mask_h = load_mask_h(file, mask_dir=mask_dir)    
    evaluation_fn = operations_d[operation.strip()]    
    result = evaluation_fn(img, segms, mask_r1, mask_r2, mask_c, mask_h, display, labels)
    return result

def evaluate_first_not_found(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels=''):
    result = evaluate_first_not_found_impl(img, segms, mask_r1, mask_r2, confusion)  
    return result

def evaluate_human_and_confusion(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels=''):
    result = evaluate_human_and_confusion_impl(img, segms, mask_r1, mask_r2, confusion, mask_h)
    return result


def evaluate_human_and_not_confusion(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels=''):   
    result = evaluate_human_and_not_confusion_impl(img, segms, mask_r1, mask_r2, confusion, mask_h)    
    return result

def evaluate_human_first_second(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels=''):
    result = evaluate_human_first_second_impl(img, segms, mask_r1, mask_r2, mask_h)

    return result

def evaluate_human(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels=''):
    result = evaluate_first_impl(img, segms, mask_h, mask_r2, confusion)      
    return result

def evaluate_first(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels=''):
    result = evaluate_first_impl(img, segms, mask_r1, mask_r2, confusion)    
    return result

def evaluate_first_second(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, plt_title='', labels=[], c_labels=''):
    result = evaluate_first_second_impl(img, segms, mask_r1, mask_r2, confusion)    
    return result

def evaluate_confusion_in_first(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels='',
                overlap_thresh=0.7):
    rslt = evaluate_confusion_in_first_impl(img, segms, mask_r1, 
                mask_r2, confusion, overlap_thresh)
    return rslt

def evaluate_confusion_not_in_first(img, segms, mask_r1, mask_r2, confusion, mask_h, display=False, labels=[], c_labels='',
                overlap_thresh=0.25):
    rslt = evaluate_confusion_not_in_first_impl(img, segms, mask_r1, 
                mask_r2, confusion, overlap_thresh)    
    return rslt
