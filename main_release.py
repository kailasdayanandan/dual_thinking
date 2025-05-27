
from operator import indexOf
import os
import shutil
import numpy as np
import pandas as pd
import glob
import collections

from os import walk
from email.mime import image
from functools import partial

import warnings
warnings.filterwarnings('ignore')

from analysis import *
from config import *

img_dir = './data/images/'
mask_dir = './data/masks/'
annotations_path = './data/DatasetFinal.csv'
file_ops_path = './data/file_ops_Final.csv'

pkl_dir = './output/model_outputs/'
results_processed_path = './output/results_all.csv'
llms_processed_path = './output/result_llms.csv'

op_plots_dir = 'output/plots'

PROCESS_IMAGE_RESULT = False #True #False #False #True
ENABLE_PLOT_ATTRIBUTES = True #False #True
SCORE_THRESHOLD = 0.7

INCLUDE_LLM_RESULTS = True #False #True

if not os.path.exists(op_plots_dir):
    os.makedirs(op_plots_dir)

##### Attributes for plotting ###########

analysis_attributes = get_dataset_attributes()

box_attributes = analysis_attributes

gestalt_attributes = ['Amodal', 'Continuity', 'Similarity', 'Proximity',  'Figure Ground']

box_plot_attributes = ['Amodal', 'Proximity', 'Similarity', 'Continuity', 'Figure Ground',\
                            'Size Diff', 'Count Diff']

table_attributes = analysis_attributes

attributes = analysis_attributes

other_attributes = analysis_attributes

model_names = get_model_name_list()

if not INCLUDE_LLM_RESULTS:
    model_names.remove('gpt-4o')
    model_names.remove('gpt-4o-mini')
    model_names.remove('llama-3.2-11B')

model_props_final = model_names

model_names_supl = model_names

#########################################

model_names_all, model_display_names_d, model_display_table_d = get_model_display_details_by_name(model_names)

if PROCESS_IMAGE_RESULT:

    processed_df = pd.read_csv(file_ops_path)
    image_files =  processed_df['filename'].to_list()

    print('Number of processed files : ', len(processed_df))
    
    evaluate_model_stats(image_files, model_names_all, processed_df, 
                op_path=results_processed_path, mask_dir=mask_dir,
                img_dir=img_dir, pkl_dir=pkl_dir)

results_df = pd.read_csv(results_processed_path)

if INCLUDE_LLM_RESULTS:

    results_llms_df = pd.read_csv(llms_processed_path)
    print(len(results_df), list(results_df))
    print(len(results_llms_df), list(results_llms_df))
    results_df = pd.concat([results_df, results_llms_df])
    print(len(results_df), list(results_df))

sub_groups_df = pd.read_csv(annotations_path)

print('-'*60)
print('SubGroup files : ', len(sub_groups_df))
print('Annotations : ', list(sub_groups_df))
print('-'*60)

model_dfs_all = get_processed_models(results_df, model_names_all, sub_groups_df)
model_dfs_all_d = dict(zip(model_names_all, model_dfs_all))

print('Model processed files : ', len(model_dfs_all[0]))

model_names = model_names_all

model_names = [x for x in model_names if x in model_names_all]

m_dfs = [model_dfs_all_d[m_name] for m_name in model_names]

results_model_attr = generate_result_attr(model_names_all, attributes, model_dfs_all_d, model_display_table_d)

model_props_final = model_names_all

model_props_final =  [x for x in model_props_final if x in model_names_all]

# --------------------------------- Model Stats  ---------------------------------

show_overall_stats(model_props_final, model_dfs_all_d)

model_stats_df = show_model_stats(model_props_final, model_display_table_d, model_dfs_all_d)
model_stack_df = model_stats_df

# --------------------------------- Model Figures  ---------------------------------

save_path = './output/plots/models_all_stacked.png'
plot_all_models_bar(save_path, model_stack_df['Model'].to_list(), model_stack_df['Correct'].to_list(), \
    model_stack_df['Human'].to_list(), model_stack_df['Error'].to_list(), 'Model Performance')


# --------------------------------- Model Table  ---------------------------------

def bold_formatter(x, value, num_decimals=2):    
    if round(x, num_decimals) == round(value, num_decimals):
        return "\\textbf{" + str(x) + "}"
    else:
        return f"{x}"

df_data = []
df_correct_data, df_human_data = [], []
for idx,r_model in enumerate(results_model_attr):
    df_row = [r_model[attr] for attr in attributes]
    df_row.insert(0, model_names_all[idx])
    df_data.append(df_row)

    df_corect_row = [r_model[attr][0] for attr in attributes]
    df_corect_row.insert(0, model_display_table_d[model_names_all[idx]])    
    df_correct_data.append(df_corect_row)

    df_human_row = [r_model[attr][1] for attr in attributes]
    df_human_row.insert(0, model_display_table_d[model_names_all[idx]])    
    df_human_data.append(df_human_row)
    
cols = attributes.copy()
cols.insert(0, 'Model')
df_stats = pd.DataFrame(data=df_data, columns=cols)

df_correct = pd.DataFrame(data=df_correct_data, columns=cols)
df_human = pd.DataFrame(data=df_human_data, columns=cols)

df_correct_ltx = df_correct[['Model'] + table_attributes]
df_human_ltx = df_human[['Model'] + table_attributes]

num_titles = []
for attr in table_attributes:
    model_attr_dfs = [mdf.loc[mdf[attr]==True] for mdf in m_dfs]    
    if attr in table_attributes:        
        ptitle = attr + ' (' + str(len(model_attr_dfs[0])) + ')'
        num_titles.append(ptitle)

num_titles = table_attributes

num_titles = list(df_correct_ltx)
num_titles.remove('Model')

table_model_list = [model_display_table_d[x] for x in model_props_final]
df_correct_ltx = df_correct_ltx[df_correct_ltx['Model'].isin(table_model_list)]
df_human_ltx = df_human_ltx[df_human_ltx['Model'].isin(table_model_list)]

def post_process_tex(latex_ip):
    latex_ip = latex_ip.replace('Figure Ground ', 'FG \\dag ')
    latex_ip = latex_ip.replace('Camouflage ', 'Camo \\ddag ')
    latex_ip = latex_ip.replace('\\end{tabular}', '{\\raggedright \\dag Figure Ground.}\\\\ \n{\\ddag Camouflage}\n\\end{tabular}')
    return latex_ip

def convert_to_tex(df_temp, cols):    
    fmts_max_2f = {column: partial(bold_formatter, value=df_temp[column].max(), num_decimals=2) for column in cols}
    fmts = dict(**fmts_max_2f)
    latex_str = df_temp.to_latex(#buf=fh,
                    index=False,
                    header=list(df_temp),
                    formatters=fmts,
                    escape=False)
    return post_process_tex(latex_str)

latex_str = convert_to_tex(df_correct_ltx, num_titles)
with open("output/gestalt_table.tex", "w") as fh:
    fh.write(latex_str)

latex_str_h = convert_to_tex(df_human_ltx, num_titles)
with open("output/gestalt_table_human.tex", "w") as fh:
    fh.write(latex_str_h)


df_dt_correct_ltx = df_correct_ltx.copy()
df_dt_correct_ltx.reset_index(drop=True, inplace=True)
df_dt_correct_ltx['Property'] = ''
df_dt_correct_ltx.loc[4,'Property'] = 'Correct'
df_dt_human_ltx = df_human_ltx.copy()
df_dt_human_ltx.reset_index(drop=True, inplace=True)
df_dt_human_ltx['Property'] = ''
df_dt_human_ltx.loc[4,'Property'] = 'Human'
df_dt_human_ltx.loc[5,0] = 'Similarity'

df_dt_correct_ltx = df_dt_correct_ltx[['Property', 'Model'] + table_attributes]
df_dt_human_ltx = df_dt_human_ltx[['Property', 'Model'] + table_attributes]

latex_str = convert_to_tex(df_dt_correct_ltx, num_titles)
latex_str_h = convert_to_tex(df_dt_human_ltx, num_titles)
latex_str = latex_str.split('\\bottomrule')[0]
latex_str_h = latex_str_h.split('\\midrule')[1]
dual_table_str = latex_str + '\\midrule\n\\midrule' + latex_str_h

with open("output/gestalt_table_both.tex", "w") as fh:
    fh.write(dual_table_str)

# --------------------------------- Attribute Box Plot  ---------------------------------

save_path = './output/plots/box.png'
box_fig = figure() 
df_box_plot = df_correct[box_plot_attributes]
boxplot = df_box_plot.boxplot(column=box_plot_attributes, grid=False, rot=45, showfliers=False)
boxplot.set_ylabel('Fraction of correct decisions')
#plt.ylim(0, 0.7)
plt.setp(boxplot.get_xticklabels(), ha="right", rotation=45)
#plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig(save_path)
plt.show()

save_path = './output/plots/human_box.png'
box_fig = figure() 
df_h_box_plot = df_human[box_plot_attributes]
boxplot = df_h_box_plot.boxplot(column=box_plot_attributes, grid=False, rot=45, showfliers=False,
                color=dict(boxes='r', whiskers='r', medians='g', caps='r'))  
boxplot.set_ylabel('Fraction of decisions similar to intuitive processing')
#plt.ylim(0, 0.7)
plt.setp(boxplot.get_xticklabels(), ha="right", rotation=45)
#plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig(save_path)
plt.show()

# --------------------------------- Model Comparison  ---------------------------------

model_names_comparison = [ 'mask_rcnn_r50', 'mask_rcnn_r101', 'mask_rcnn_x101', 
                        #'instaboost_r50', 'instaboost_r101', 'instaboost_x101',
                        #'groie_r50', 'groie_r101',                        
                        #'yolact_r50', 'yolact_r101', 
                        'swin-s',  'swin-t',
                        'gpt-4o', 'gpt-4o-mini', 'llama-3.2-11B', 'llama-3.2-90B',
                        ]

model_names_comparison =  [x for x in model_names_comparison if x in model_names_all] 

comparison_attributes = ['Size Diff','Count Diff','Camouflage',        
                        'Figure Ground', 'Amodal', 'Similarity']

process_data_box(model_names_comparison, results_model_attr,    
                    model_names_all, model_display_names_d, 
                    comparison_attributes, './output/plots/comparison_prop_plot.png')

df_stats = pd.DataFrame(data=df_data, columns=cols)

df_correct = pd.DataFrame(data=df_correct_data, columns=cols)
df_human = pd.DataFrame(data=df_human_data, columns=cols)

df_correct_ltx = df_correct[['Model'] + comparison_attributes]
df_human_ltx = df_human[['Model'] + comparison_attributes]

num_titles = []
for attr in comparison_attributes:
    model_attr_dfs = [mdf.loc[mdf[attr]==True] for mdf in m_dfs]    
    if attr in comparison_attributes:        
        ptitle = attr + ' (' + str(len(model_attr_dfs[0])) + ')'
        num_titles.append(ptitle)

num_titles = comparison_attributes

num_titles = list(df_correct_ltx)
num_titles.remove('Model')

table_model_list = [model_display_table_d[x] for x in model_names_comparison]
df_correct_ltx = df_correct_ltx[df_correct_ltx['Model'].isin(table_model_list)]
df_human_ltx = df_human_ltx[df_human_ltx['Model'].isin(table_model_list)]

df_dt_correct_ltx = df_correct_ltx.copy()
df_dt_correct_ltx.reset_index(drop=True, inplace=True)
df_dt_correct_ltx['Property'] = ''
df_dt_correct_ltx.loc[4,'Property'] = 'Correct'
df_dt_human_ltx = df_human_ltx.copy()
df_dt_human_ltx.reset_index(drop=True, inplace=True)
df_dt_human_ltx['Property'] = ''
df_dt_human_ltx.loc[4,'Property'] = 'Human'
df_dt_human_ltx.loc[5,0] = 'Similarity'

df_dt_correct_ltx = df_dt_correct_ltx[['Property', 'Model'] + comparison_attributes]
df_dt_human_ltx = df_dt_human_ltx[['Property', 'Model'] + comparison_attributes]

latex_str = convert_to_tex(df_dt_correct_ltx, num_titles)
latex_str_h = convert_to_tex(df_dt_human_ltx, num_titles)
latex_str = latex_str.split('\\bottomrule')[0]
latex_str_h = latex_str_h.split('\\midrule')[1]
dual_table_str = latex_str + '\\midrule\n\\midrule' + latex_str_h

with open("output/comparison_table_both.tex", "w") as fh:
    fh.write(dual_table_str)




    
