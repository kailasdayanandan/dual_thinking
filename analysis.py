
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

from textwrap import wrap
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter

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

        #if model_name == 'mask_rcnn_r50' or model_name == 'mask_rcnn_r101':
        #    continue

        picke_path_gz = os.path.join(pkl_dir,model_name)
        picke_path_gz = picke_path_gz + '.zip'

        with gzip.open(picke_path_gz, 'rb') as pickle_file:
            results_dict = cPickle.load(pickle_file)
        
        #error_files = []
        for fname in tqdm(i_files):            
            if True : #try:
                correct_ops, human_ops = get_all_ops_defined(fname, p_df)    
                _, labels, segms, _ = results_dict[fname]
                img_path = img_dir + fname
                img = Image.open(img_path)
                result_h = evaluate_operation(fname, 'Human', human_ops, segms, img, labels=labels, mask_dir=mask_dir)
                result = evaluate_operation(fname, 'Correct', correct_ops, segms, img, labels=labels, mask_dir=mask_dir)
                error = (not result) and (not result_h)
                results_p.append((model_name, fname, result, result_h, error))        
            #except:
            #    print(fname)
            #    error_files.append(error_files)

        #print(error_files)
        #break


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

def get_colors_plot(num=100):    
    d_colors_temp = [
            (0.2549019607843137, 0.35294117647058826, 0.5490196078431373),
            (0.7235294117647058, 0.1882352941176471, 0.0),
            (0.6470588235294118, 0.11764705882352941, 0.21568627450980393),
            (0.23137254901960786, 0.23137254901960786, 0.13137254901960786),
            (0.2549019607843137, 0.35294117647058826, 0.5490196078431373),
            (0.8235294117647058, 0.5882352941176471, 0.3),
            (0.6470588235294118, 0.11764705882352941, 0.21568627450980393),                        
            #(0.3, 0.4882352941176471, 0.8235294117647058),            
            (0.0, 0.4882352941176471, 0.8235294117647058),            
            #(0.8235294117647058, 0.5882352941176471, 0.3),
            #(0.43137254901960786, 0.43137254901960786, 0.43137254901960786),
            #(0.6470588235294118, 0.11764705882352941, 0.21568627450980393),            
            (0.2549019607843137, 0.35294117647058826, 0.5490196078431373),
            (0.8235294117647058, 0.5882352941176471, 0.0),
            (0.6470588235294118, 0.11764705882352941, 0.21568627450980393),
            (0.43137254901960786, 0.43137254901960786, 0.43137254901960786),
            (0.2549019607843137, 0.35294117647058826, 0.5490196078431373),
            (0.8235294117647058, 0.5882352941176471, 0.0),
            (0.6470588235294118, 0.11764705882352941, 0.21568627450980393),
            (0.43137254901960786, 0.43137254901960786, 0.43137254901960786)
        ]
    if num < len(d_colors_temp):
        #print("USING STATIC COLORS..")
        return d_colors_temp
    colrs = [(random.uniform(0, 0.85),random.uniform(0, 0.9),random.uniform(0, 0.8)) for i in range(num)]
    return colrs

def plot_properties(class_vals, d_colors, markers, 
                    p_names, classes, xs_label, show_labels=False,
                    figure_path = './output/final.png'):
    
    ICONS_DIR = "./assets/icons_human/" #"./assets/icons/"
    #figure_path = './output/shape_bias_dummy.png'

    # global default boundary settings for thin gray transparent
    # boundaries to avoid not being able to see the difference
    # between two partially overlapping datapoints of the same color:
    PLOTTING_EDGE_COLOR = (0.3, 0.3, 0.3, 0.3)
    PLOTTING_EDGE_WIDTH = 0.02

    fontsize = 15 #20 #25
    ticklength = 3 #5 #10
    markersize = 250

    if show_labels:
        fontsize = 10
        markersize = 150

    num_classes = len(classes)
    
    # plot setup
    if show_labels:
        wdth = 11 #9
        #if len(classes) <= 5:
        #    ht = 4
        if len(classes) < 7:
            ht = 5
        elif len(classes) < 10:
            ht = 7
        else :
            ht = 10
        fig = plt.figure(1, figsize=(wdth, ht), dpi=100.)
    else :
        fig = plt.figure(1, figsize=(7, 7), dpi=100.)

    classes = ['\n'.join(wrap(x, 14)) for x in  classes]

    ax = plt.gca()

    ax.set_xlim([0, 1])
    #if not show_labels:
    ax.set_ylim([-.5, num_classes - 0.5])

    # secondary reversed x axis
    #ax_top = ax.secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))

    # labels, ticks
    if not show_labels:
        plt.tick_params(axis='y',
                    which='both',
                    left=False,
                    right=False,
                    labelleft=False)
    #ax.set_ylabel("Properies", labelpad=60, fontsize=fontsize)
    ax.set_xlabel(xs_label, fontsize=fontsize, labelpad=10)
    #ax_top.set_xlabel("Fraction of 'shape' decisions", fontsize=fontsize, labelpad=25)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    #ax_top.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.get_xaxis().set_ticks(np.arange(0, 1.1, 0.1))
    if show_labels:
        #print('--- Classes : ', classes)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)        
        #ax.yaxis.set_ticks(classes)        
    #ax_top.set_ticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize, length=ticklength)
    #ax_top.tick_params(axis='both', which='major', labelsize=fontsize, length=ticklength)

    # icons besides y axis
    # icon placement is calculated in axis coordinates
    WIDTH = 1 / num_classes  #
    XPOS = -1.0 * WIDTH  # placement left of yaxis (-WIDTH) plus some spacing (-.25*WIDTH)
    YPOS = -0.5
    HEIGHT = 1
    MARGINX = 1 / 10 * WIDTH 
    MARGINY = 1 / 10 * HEIGHT 

    fac_m = 1 #2
    left = XPOS + (MARGINX * fac_m)
    right = XPOS + WIDTH - (MARGINX * fac_m)

    if not show_labels:
        for i in range(num_classes):
            bottom = i + MARGINY + YPOS
            top = (i + 1) - MARGINY + YPOS
            iconpath = pjoin(ICONS_DIR, "{}.png".format(classes[i].lower()))
            img = plt.imread(iconpath)            
            img = frame_image(img, 7)
            #print(i, [left, right, bottom, top])        
            plt.imshow(img, extent=[left, right, bottom, top], aspect='auto', clip_on=False)

    # plot horizontal intersection lines
    for i in range(num_classes - 1):
        plt.plot([0, 1], [i + .5, i + .5], c='gray', linestyle='dotted', alpha=0.4)

    # plot average shapebias + scatter points
    #print(len(class_vals[0]), len(classes), classes)
    
    for i in range(len(class_vals)):        
        ax.scatter(class_vals[i][0:len(classes)], classes,
                color=d_colors[i],
                marker=markers[i],
                label=p_names[i],
                s=markersize,
                clip_on=False,
                edgecolors=PLOTTING_EDGE_COLOR,
                linewidths=PLOTTING_EDGE_WIDTH,
                zorder=3)
        ax.legend()


    #figure_path = pjoin(result_dir, f"{ds.name}_shape-bias_matrixplot.pdf")    
    fig.savefig(figure_path) #, bbox_inches='tight')
    plt.show()
    plt.close()

def get_color_plot(model_names):
    colrs = []
    for model_name in model_names:
        if 'r50' in model_name.lower():
            colrs.append((0.6470588235294118, 0.11764705882352941, 0.21568627450980393))
        elif 'r101' in model_name.lower():
            #colrs.append((0.23137254901960786, 0.23137254901960786, 0.13137254901960786))
            colrs.append((0.0705882352941176, 0.4431372549019608, 0.1098039215686275))
        elif 'x101' in model_name.lower():
            #colrs.append((1, 0.85, 0))
            #colrs.append((0.11764705882352941, 0.21568627450980393, 0.6470588235294118))
            colrs.append((0.0235294117647059, 0.2235294117647059, 0.4392156862745098))
    return colrs
    

def process_data_box(model_names, results_model_attr, model_names_all, model_display_names_d, 
        attribs, figure_path, correct=True):

    print('model_names : ', len(model_names))
    print(model_names)

    ex_results = extract_models_result(model_names, results_model_attr, model_names_all)

    attr_plot = attribs
    
    class_correct = [ [x[attr][0] for attr in attr_plot] for x in ex_results]
    class_human = [ [x[attr][1] for attr in attr_plot] for x in ex_results]

    if correct:
        class_val = class_correct
        xlabel = "Fraction of correct decisions"
    else :
        class_val = class_human
        xlabel = "Fraction of decisions similar to human"

    markers = []
    for x in model_names:
        if ('swin' in x.lower()) or ('former' in x.lower()) or ('query' in x.lower()):
        #if x.startswith('groie'): 
            markers.append('*')
        elif x.startswith('mask_rcnn'): #simple-copy-paste'): 
            markers.append('X') #P')
        elif x.startswith('insta'): 
            markers.append('P')
        #elif x.startswith('detectors_htc_r50'): 
        #    markers.append('d')
        elif x.startswith('gestalt_r50'): 
            markers.append('D')
        elif x.startswith('rfp_htc_r50'): 
            markers.append('s')
        elif x.startswith('sac_htc_r50'): 
            markers.append('X')
        else :
            markers.append('o')

    display_names = [model_display_names_d[x] for x in model_names]

    d_colors_temp = get_colors_plot(len(display_names))
    #d_colors_temp = get_color_plot(model_names)

    
    print('display_names : ', len(display_names))
    print(display_names)

    plot_properties(class_val, d_colors_temp, markers, display_names, attr_plot, 
        xlabel, show_labels=True, figure_path = figure_path)# attr_plot)
    
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
