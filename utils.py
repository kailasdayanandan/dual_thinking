
import os
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

def load_mask(file, mask_file, mask_dir='./data/masks/'):
    img_dir = file.rstrip('.jpg')
    img_dir = file[:-4]
    folder = mask_dir + img_dir + '/'
    r1_path = folder + mask_file
    if not os.path.exists(r1_path):
        return None
    img = Image.open(r1_path).convert('L')    
    return np.asarray(img)

def load_mask_r1(file, mask_dir='./data/masks/'):
    return load_mask(file, 'mask_r1.png', mask_dir=mask_dir)

def load_mask_r2(file, mask_dir='./data/masks/'):
    return load_mask(file, 'mask_r2.png', mask_dir=mask_dir)

def load_mask_c(file, mask_dir='./data/masks/'):
    return load_mask(file, 'mask_c.png', mask_dir=mask_dir)

def load_mask_h(file, mask_dir='./data/masks/'):
    return load_mask(file, 'mask_h.png', mask_dir=mask_dir)


def get_masked_region(img, mask):
    extractedregion = (img.T * mask.T).T
    return extractedregion

def get_equal_result(df, models):
    eq_df = df[df[models].apply(pd.Series.nunique, axis=1) == 1]
    return eq_df

def get_not_equal_result(df,models):
    eq_df = df[df[models].apply(pd.Series.nunique, axis=1) != 1]
    return eq_df

def get_equal_result(df, models):
    eq_df = df[df[models].apply(pd.Series.nunique, axis=1) == 1]
    return eq_df

def get_not_equal_result(df, models):
    eq_df = df[df[models].apply(pd.Series.nunique, axis=1) != 1]
    return eq_df