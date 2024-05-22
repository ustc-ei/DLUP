from multiprocessing.context import SpawnContext
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import scipy
from scipy import sparse
import random
import os
import collections
from numba import jit
import sys
from matplotlib_venn import venn2
from matplotlib import pyplot as plt
import pandas as pd
import idpicker

def protein_id(pep_id):
    pep_intensity = pd.read_csv(
        "/data/dg/CNNp/lib/AD8-300S-directDIA.xls", sep='\t')
    pep_intensity_filter = pep_intensity[[
        "ModifiedPeptide", "PrecursorCharge" ,"Protein Name"]].values
    dict_temp = dict()
    for temp in pep_intensity_filter:
        dict_temp[(temp[0],temp[1])] = temp[2].split(';')
    tuple_list=[]
    for i in pep_id:
        for j in dict_temp[i]:
            tuple_list.append((i,j))
    return idpicker.find_valid_proteins(tuple_list)

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/all_9plasma_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "7000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/all_9plasma_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/all_9plasma_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"

# ]

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/8of20plasma_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "9000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/8of20plasma_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/8of20plasma_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"

# ]

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/20_CCS_plasma_QC_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "10000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/20_CCS_plasma_QC_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/20_CCS_plasma_QC_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"
    
# ]

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/1-24_CCS_plasma_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "10000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/1-24_CCS_plasma_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/1-24_CCS_plasma_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"
    
# ]

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/rep_plasma_25QC_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "10000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/rep_plasma_25QC_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/rep_plasma_25QC_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"
    
# ]

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/PXD040205_technicalQC_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "7000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/PXD040205_technicalQC_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/PXD040205_technicalQC_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"
    
# ]

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/PXD040205_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "7000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/PXD040205_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/PXD040205_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"
    
# ]

# args=[
#     "",
#     "/data/dg/myNetwork/quant/quant_result/20231018_dataset_PXD036608_oldlib_mobility_window_data_process/",
#     "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
#     "1",
#     "0.000001",
#     "8000",
#     "64",
#     "/data/dg/myNetwork/identify/identify_result/20231018_dataset_PXD036608_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
#     "/data/dg/myNetwork/identify/identify_result/20231018_dataset_PXD036608_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"
    
# ]

args=[
    "",
    "/data/dg/myNetwork/quant/quant_result/20231018_dataset_PXD030327_repeat3_oldlib_mobility_window_data_process/",
    "good_unique_new_plasma_1-20_mobility_d_train_win_data_minmz3_Spectronaut_cv_ab_ratio_0.5_quant_min_test_loss",
    "1",
    "0.000001",
    "7000",
    "64",
    "/data/dg/myNetwork/identify/identify_result/20231018_dataset_PXD030327_repeat3_oldlib_mobility_window_data_process/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy",
    "/data/dg/myNetwork/identify/identify_result/20231018_dataset_PXD030327_repeat3_oldlib_mobility_window_data_process_decoy/paper_new_pa_1-20_20230221_pa_1-20_pa_1-9_20230221_pa_1-24_rep_pa_25QC_mobility_identify_max_test_true_learning_rate_0.0002_range_time_300_batch_size_4096.npy"
]

quant_dir_path = args[1]  # 定量结果目录（不包含文件名，包含/）
quant_file_name = args[2]  # 定量结果文件名（不包含_learning_*.npy的文件名）
ifID = int(args[3])  # 是否进行定性过滤，1表示是，0表示不是
learning_rate = float(args[4])
range_time = int(args[5])
batch_size = int(args[6])
if len(args) > 7 and ifID != 0:
    identify_path = args[7]  # 定性文件路径，包含文件名
if len(args) > 8 and ifID != 0:
    identifyd_path = args[8]  # 定性文件路径，包含文件名
if len(args) > 9:
    range_time = args[9]

protein_csv_name="deep_learning_dataset-PXD030327_repeat3_protein_Report.tsv"

root_path = '/data/dg/CNNp/lib/'
library = np.load(
    root_path + "AD8-300S-directDIA_ModifiedPeptide_DDA_library_im_norm_peak6.npy", allow_pickle=True).item()
for key in tqdm(library.keys()):
    library[key]['Species'] = 'HUMAN'
# 修正肽段名不同导致的错误
pep_intensity = pd.read_csv(
    "/data/dg/CNNp/lib/AD8-300S-directDIA.xls", sep='\t')
pep_intensity_filter = pep_intensity[[
    "ModifiedPeptide", "StrippedPeptide"]].values
StrippedPep2ModifiedPep = dict()
for temp in pep_intensity_filter:
    StrippedPep2ModifiedPep[temp[1]] = temp[0]
ModifiedPep2StrippedPep = dict()
for temp in pep_intensity_filter:
    ModifiedPep2StrippedPep[temp[0]] = temp[1]
# 修正肽段名不同导致的错误

quant_file_name = quant_file_name+"_learning_rate_" + \
    str(learning_rate)+"_range_time_"+str(range_time) + \
    "_batch_size_"+str(batch_size)+".npy"
quant_path = os.path.join(quant_dir_path, quant_file_name)
quant_file = np.load(quant_path, allow_pickle=True)
writer_file_name = quant_path.split('/')[-2]+'_'+quant_path.split(
    '/')[-1].split('_learning_rate_')[0]+'_ifID_'+str(ifID)+'.txt'  # 记录结果的文件名
dirname1 = quant_path.split(
    '/')[-2]+'_'+quant_path.split('/')[-1].split('_learning_rate_')[0]+'_ifID_'+str(ifID)
dirname1 = "quant_eval/"+dirname1
# if os.path.exists(dirname1) == False:
#     os.mkdir(dirname1)

file_seq_charge, quant_all = quant_file[0], quant_file[1]

filenames=list(set([i[0] for i in file_seq_charge]))
filenames=sorted(filenames)

identify = np.load(identify_path, allow_pickle=True)
identifyd = np.load(identifyd_path, allow_pickle=True)
# deep identify
ip = np.array([np.array(i) for i in identify[1]])
t = ip[:, 0]
t.sort()

dp = np.array([np.array(i) for i in identifyd[1]])
td = dp[:, 0]
td.sort()

thresh = 1000
d_i = 0
for i in tqdm(range(len(t))):
    while td[d_i] < t[i]:
        d_i += 1
        if d_i == len(td):
            break
    if d_i == len(td):
        break
    fdr = (len(td)-d_i)/(len(t)-i)
    if fdr < 0.01:
        thresh = t[i]
        break
print(thresh, fdr, i, d_i, len(td[td >= thresh]), len(t[t >= thresh]))

deep_identify = {i: [] for i in filenames}
for seq_i in range(len(identify[0])):
    if identify[1][seq_i][0] >= thresh:
        if identify[0][seq_i][0] in deep_identify.keys():
            deep_identify[identify[0][seq_i][0]].append(identify[0][seq_i][1])
print(sum([len(v) for v in deep_identify.values()]))
print({k: len(v) for k, v in deep_identify.items()})

for k, v in deep_identify.items():
    deep_identify[k] = list(set(v))
print(sum([len(v) for v in deep_identify.values()]))
print({k: len(v) for k, v in deep_identify.items()})

names = {i: [] for i in filenames}
for seq_i in tqdm(range(len(file_seq_charge))):
    t_file = file_seq_charge[seq_i][0]
    if t_file in deep_identify.keys():
        if file_seq_charge[seq_i][1] not in deep_identify[t_file]:
            continue
    t_seq = file_seq_charge[seq_i][1][0]
    t_charge = file_seq_charge[seq_i][1][1]
    # t_species = 'HUMAN'
    # 这里修改为统一格式
    t_species = library[file_seq_charge[seq_i][1]]['Species']
    if t_file in names.keys():
        names[t_file].append([t_seq, t_charge, quant_all[seq_i][0], t_species])

names2={i: dict() for i in filenames}
for i in filenames:
    for j in names[i]:
        names2[i][ModifiedPep2StrippedPep[j[0]]]=0
for i in filenames:
    for j in names[i]:
        names2[i][ModifiedPep2StrippedPep[j[0]]]+=j[2]

all_pep_list=[]
for i in filenames:
    all_pep_list.extend(list(names2[i].keys()))
all_pep_list=sorted(list(set(all_pep_list)))
column_list=["PEP.StrippedSequence"]
for i in range(len(filenames)):
    column_list.append("["+str(i+1)+"] "+str(filenames[i])+".d.PEP.Quantity")
csv_data=dict()
for i in column_list:
    csv_data[i]=[]
csv_data[column_list[0]]=all_pep_list
for i in range(len(all_pep_list)):
    for j in range(len(filenames)):
        if all_pep_list[i] in names2[filenames[j]].keys():
            csv_data[column_list[j+1]].append(names2[filenames[j]][all_pep_list[i]])
        else:
            csv_data[column_list[j+1]].append(float('nan'))

csv_data_df = pd.DataFrame(csv_data)
# 设置保存为CSV时处理NaN的选项
options = {
    'na_rep': 'NaN',   # 指定NaN在CSV中显示为'NaN'
    'index': False  # 不保存索引列到CSV文件中
}
# 将DataFrame写入CSV文件
# csv_data_df.to_csv('peptide_output2.csv', **options)

names3={i: dict() for i in filenames}
for i in filenames:
    for j in names[i]:
        names3[i][(j[0],j[1])]=j[2]

pep_intensity = pd.read_csv(
    "/data/dg/CNNp/lib/AD8-300S-directDIA.xls", sep='\t')
pep_intensity_filter = pep_intensity[[
    "ModifiedPeptide", "PrecursorCharge" ,"Protein Name"]].values
pep_charge2protein = dict()
for temp in pep_intensity_filter:
    pep_charge2protein[(temp[0],temp[1])] = temp[2].split(';')
protein2pep_charge = dict()
for key,value in pep_charge2protein.items():
    for v in value:
        protein2pep_charge[v]=[]
for key,value in pep_charge2protein.items():
    for v in value:
        protein2pep_charge[v].append(key)

protein_quant_result={i: dict() for i in filenames}
for i in filenames:
    pep_list=deep_identify[i]
    protein_group_list=list(set(protein_id(pep_list).values()))
    #对同一个group的蛋白质进行排序
    protein_group_list=[temp.split("/")[0]+'/'+'/'.join(sorted(temp.split("/")[1:])) for temp in protein_group_list]
    for j in protein_group_list:
        j_split=j.split("/")
        k=j_split[1]
        protein_quantity=np.sum([names3[i][p] for p in protein2pep_charge[k] if p in names3[i].keys()])
        protein_quant_result[i][';'.join(j_split[1:])]=protein_quantity      
all_protein_list=[]
for i in filenames:
    all_protein_list.extend(list(protein_quant_result[i].keys()))
all_protein_list=sorted(list(set(all_protein_list)))
column_list=["PG.ProteinNames"]
for i in range(len(filenames)):
    column_list.append("["+str(i+1)+"] "+str(filenames[i])+".d.PG.Quantity")
csv_data=dict()
for i in column_list:
    csv_data[i]=[]
csv_data[column_list[0]]=all_protein_list
for i in range(len(all_protein_list)):
    for j in range(len(filenames)):
        if all_protein_list[i] in protein_quant_result[filenames[j]].keys():
            csv_data[column_list[j+1]].append(protein_quant_result[filenames[j]][all_protein_list[i]])
        else:
            csv_data[column_list[j+1]].append(float('nan'))

csv_data_df = pd.DataFrame(csv_data)
# 设置保存为CSV时处理NaN的选项
options = {
    'sep': '\t',
    'na_rep': 'NaN',   # 指定NaN在CSV中显示为'NaN'
    'index': False  # 不保存索引列到CSV文件中
}
# 将DataFrame写入CSV文件
csv_data_df.to_csv(protein_csv_name, **options)

