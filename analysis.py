
import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import _pickle as cPickle
import gzip


from automate import *


#-----------------------------------------------------------------------------------

def show_overall_stats(model_props_finals, m_dfs_d):   

    m_dfs = [m_dfs_d[model] for model in model_props_finals]    
    print('-'*150)
    print('Using ', len(m_dfs), ' of ', len(m_dfs_d), ' models.')
    correct = [m_dfs[i]['Correct'] for i in range(len(m_dfs))]
    correct = np.sum(np.stack(correct),axis=0)
    print('All models predicted correctly : ', np.sum(correct == len(m_dfs)), ' \t\tPercentage : ', \
        "{:.2f}".format((100 * np.sum(correct == len(m_dfs))) / len(m_dfs[0])), '%')
    print('All models predicted wrongly : ', np.sum(correct == 0), \
        ' \t\tPercentage : ', "{:.2f}".format((100 * np.sum(correct == 0)) / len(m_dfs[0])), '%')

    human = [m_dfs[i]['Human'] for i in range(len(m_dfs))]
    human = np.sum(np.stack(human),axis=0)
    print('All models predicted like human : ', np.sum(human == len(m_dfs)), \
        ' \t\tPercentage : ', "{:.2f}".format((100 * np.sum(human == len(m_dfs))) / len(m_dfs[0])), '%')
    print('-'*150)

def show_model_stats(m_names, m_table_d, m_dfs_d):
     
    model_stats = []
    
    for model in m_names:    
        t_df = m_dfs_d[model]
        total = len(t_df)
        human_p = round(np.sum(t_df['Human'])/ total, 2)
        correct_p = round(np.sum(t_df['Correct'])/ total, 2)
        error_p = round(np.sum(t_df['Error'])/ total, 2)
        hbias_p = np.array(human_p, dtype='float32')/np.array(correct_p, dtype='float32')
        model_stats.append((m_table_d[model], correct_p, human_p, error_p, hbias_p))

    model_stats_df = pd.DataFrame(model_stats, columns=['Model', 'Correct', 'Human', 'Error', 'HBias'])
    print(model_stats_df.head(len(model_stats_df)))

    print('-'*10, 'Sorted on Correct', '-'*10)
    model_stats_df = model_stats_df.sort_values('Correct', ascending=False)
    model_stats_df.reset_index(drop=True, inplace=True)
    print(model_stats_df.head(len(model_stats_df)))

    print('-'*10, 'Sorted on Human', '-'*10)
    model_stats_df = model_stats_df.sort_values('Human', ascending=False)
    model_stats_df.reset_index(drop=True, inplace=True)
    print(model_stats_df.head(len(model_stats_df)))

    print('-'*10, 'Sorted on HBias', '-'*10)
    model_stats_df = model_stats_df.sort_values('HBias', ascending=False)
    model_stats_df.reset_index(drop=True, inplace=True)
    print(model_stats_df.head(len(model_stats_df)))

    print('-'*150)

    return model_stats_df.sort_values('Correct', ascending=False)

#-----------------------------------------------------------------------------------

def get_processed_models(r_df, m_names, groups_df):
    model_dfs = []
    for model_name in m_names:    
        model_df = r_df[r_df['model']== model_name]
        model_df = pd.merge(model_df, groups_df, on="filename")
        model_dfs.append(model_df)
    return model_dfs

#-----------------------------------------------------------------------------------

def get_all_ops_defined(fname, ops_df):
    if fname in ops_df['filename'].to_list():
        eval_op = ops_df.loc[ops_df.filename == fname, 'Correct'].item()
        eval_op_h = ops_df.loc[ops_df.filename == fname, 'Human'].item()
        return eval_op, eval_op_h
    return None, None

def evaluate_model_stats(i_files, m_names, p_df, op_path='./output/results_all.csv', 
            img_dir = './data/images/', mask_dir='./data/masks/',
            pkl_dir = './output/model_outputs/'):
    
    results_p = []

    for model_name in m_names:    

        print('-'*60, model_name,'-'*60)

        picke_path_gz = os.path.join(pkl_dir,model_name)
        picke_path_gz = picke_path_gz + '.zip'

        with gzip.open(picke_path_gz, 'rb') as pickle_file:
            results_dict = cPickle.load(pickle_file)
        
        for fname in tqdm(i_files):
            correct_ops, human_ops = get_all_ops_defined(fname, p_df)    
            _, labels, segms, _ = results_dict[fname]
            img_path = img_dir + fname
            img = Image.open(img_path)
            result_h = evaluate_operation(fname, 'Human', human_ops, segms, img, labels=labels, mask_dir=mask_dir)
            result = evaluate_operation(fname, 'Correct', correct_ops, segms, img, labels=labels, mask_dir=mask_dir)
            error = (not result) and (not result_h)
            results_p.append((model_name, fname, result, result_h, error))        

    results_df = pd.DataFrame(results_p, columns=['model', 'filename', 'Correct', 'Human', 'Error'])
    print(len(results_df))
    results_df.to_csv(op_path, index=False)

#-----------------------------------------------------------------------------------

def generate_result_attr(m_names, attributes, model_dfs_d, m_table_d):
    results_model_attr = [dict() for x in range(len(m_names))]
    m_dfs = [model_dfs_d[m_name] for m_name in m_names]
    
    for attr in attributes:        
        model_attr_dfs = [mdf.loc[mdf[attr]==True] for mdf in m_dfs]
        #print('-'*60, attr + ' (' + str(len(model_attr_dfs[0])) + ')', '-'*60)
        #ptitle = attr + ' (' + str(len(model_attr_dfs[0])) + ')'
        print('='*60, attr, '='*60)
        #ptitle = attr 
        y_human = [np.sum(model_attr_dfs[i]['Human']) for i in range(len(model_attr_dfs))]
        y_correct = [np.sum(model_attr_dfs[i]['Correct']) for i in range(len(model_attr_dfs))]
        y_error = [np.sum(model_attr_dfs[i]['Error']) for i in range(len(model_attr_dfs))]

        total = len(model_attr_dfs[0])
        model_stats = []            
        for i,model in enumerate(m_names):    
            human_p = round(y_human[i]/ total, 2)
            #human_s = "{:.2f}".format(human_p)
            correct_p = round(y_correct[i]/ total, 2)
            #correct_s = "{:.2f}".format(correct_p)
            error_p = round(y_error[i]/ total, 2)
            #error_s = "{:.2f}".format(error_p)
            model_stats.append((m_table_d[model], correct_p, human_p, error_p))
            results_model_attr[i][attr] = (correct_p, human_p, error_p)

        model_stats_df = pd.DataFrame(model_stats, columns=['Model', 'Correct', 'Human', 'Error'])
        model_stats_df = model_stats_df.sort_values('Correct', ascending=False)
        print(model_stats_df.head(len(model_stats_df)))

        print('-'*10, 'Sorted on Correct', '-'*10)
        model_stats_df = model_stats_df.sort_values('Correct', ascending=False)
        model_stats_df.reset_index(drop=True, inplace=True)
        print(model_stats_df.head(len(model_stats_df)))

        print('-'*10, 'Sorted on Human', '-'*10)
        model_stats_df = model_stats_df.sort_values('Human', ascending=False)
        model_stats_df.reset_index(drop=True, inplace=True)
        print(model_stats_df.head(len(model_stats_df)))

        hbias_p = np.array(model_stats_df['Human'], dtype='float32')/np.array(model_stats_df['Correct'], dtype='float32')
        model_stats_df['HBias'] = hbias_p
        print('-'*10, 'Sorted on HBias', '-'*10)
        model_stats_df = model_stats_df.sort_values('HBias', ascending=False)
        model_stats_df.reset_index(drop=True, inplace=True)
        print(model_stats_df.head(len(model_stats_df)))

    return results_model_attr


def extract_models_result(models, results_all, model_names_all):
    ex_results = [results_all[model_names_all.index(model)] for model in models]
    return ex_results
    
#-----------------------------------------------------------------------------------

def plot_all_models_bar(path, model_names, correct_values, human_values, other_values, ylabel=None,
            figsize=(10, 6), ylim=None):

    color_correct = (150 / 255.0, 210 / 255.0, 150 / 255.0)
    color_human = (213 / 255.0, 150 / 255.0, 150 / 255.0)
    color_other = (215 / 255.0, 215 / 255.0, 215 / 255.0)
    
    accuracy_counts = {
        "Correct Perception": correct_values,
        "Intuitive Perception": human_values,
        "Other Errors": other_values,
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    for s in ["right", "top"]:  # , "bottom"]:
        ax.spines[s].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels([])
    if ylim is None:
        limit = np.ceil(max(correct_values) * 10) / 10.0
        ylim = (min(0, min(correct_values)),
                max(0, limit))
    plt.ylabel(ylabel, fontsize=15)
        
    bottom = np.zeros(len(model_names))
    index = 0
    for ky, acc_count in accuracy_counts.items():   
        if 'Correct' in ky:
            color_t = color_correct
        elif 'Intuitive' in ky:
            color_t = color_human
        elif 'Other' in ky:
            color_t = color_other

        p = ax.bar(model_names, acc_count, width=0.5, label=ky, bottom=bottom, color=color_t) 
        index += 1
        bottom += acc_count
    
    for i, rect in enumerate(p):
        plt.text(rect.get_x() + rect.get_width() / 2.0, 0.01 * limit,
                    model_names[i], ha='center',
                    va='bottom', rotation=90)

    ax.legend(loc="upper right")
    plt.savefig(path)
    plt.show()
