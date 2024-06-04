
import os
import numpy as np
import torch
import wget
from tqdm import tqdm
import mmcv
from mmdet.core import get_classes
from mmdet.apis import init_detector, inference_detector

import gzip
import _pickle as cPickle


from os import walk
from config import get_mmdetection_model_config,get_model_name_list

def proces_result(result_s, score_thr=0.7):

    bbox_result, segm_result = result_s    
    bboxes = np.vstack(bbox_result)
    labels = [ np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result) ]
    labels = np.concatenate(labels)

    if len(labels) == 0:
        return [], [], [], []

    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    score_thresh = scores[inds]
    if segms is not None:
        segms = segms[inds, ...]
        
    class_names = get_classes('coco')
    labels =  [class_names[i] for i in labels]
    return bboxes, labels, segms, score_thresh

def load_model(mdl_cfg, mdl_p_url):
    model = init_detector(mdl_cfg, mdl_p_url, device='cuda:0')
    return model

def mm_model_detect(model_name, filenames, image_dir, pretrained_dir, op_dir = '', force_generate=False):
    print('Processing : ', model_name)
    picke_path_gz = os.path.join(op_dir,model_name)
    picke_path_gz = picke_path_gz + '.zip'
    if os.path.exists(picke_path_gz) and not force_generate:
        print('Loading from pickle file : ', picke_path_gz)
        with gzip.open(picke_path_gz, 'rb') as pickle_file:
            result = cPickle.load(pickle_file)
        return result, picke_path_gz

    mm_config, mm_weights, mm_url = get_mmdetection_model_config(model_name)
    mm_weights_path = os.path.join(pretrained_dir,mm_weights)
    if not os.path.exists(mm_weights_path):
        response = wget.download(mm_url, mm_weights_path)
        print('Downloading weights .. ', response)

    model = load_model(mm_config, mm_weights_path)
    results_dict = dict()
    for fname in tqdm(filenames):        
        img_path = os.path.join(image_dir,fname)
        result = inference_detector(model, img_path)    
        bboxes, labels, segms, score_thresh = proces_result(result)
        results_dict[fname] = (bboxes, labels, segms, score_thresh)

    
    print(picke_path_gz, picke_path_gz)
    #print('Checking before saving ... ')
    #proces_result(result, score_thr=0.7)
    print('\t Saving to : ', picke_path_gz)
    with gzip.open(picke_path_gz, 'wb') as pickle_file:
        cPickle.dump(results_dict, pickle_file)

    return results_dict, picke_path_gz

def mm_model_load_segment(results_dict, filename):
    bboxes, labels, segms, score_thresh = results_dict[filename]
    return bboxes, labels, segms, score_thresh
        
if __name__ == '__main__':

    img_dir = './data/images'
    image_files = filenames = next(walk(img_dir), (None, None, []))[2] 
    pretrained_dir = '/home/dell/kailasd/2024_final/output/pretrained/'
    op_dir = 'output/model_outputs/'

    models = get_model_name_list()

    for model_name in models:
        result = mm_model_detect(model_name, image_files, img_dir, pretrained_dir, op_dir = op_dir, force_generate=True)


