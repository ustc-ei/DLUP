import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import List, Set, Dict
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import idpicker

def plot_venn(deep_learning: List, spectronaut: List, overlaps: List, data_filename: List[str], is_train: bool, fig_title: str, save_rootpath: str, is_all_cup=False):
    plt.rcParams['font.family'] = ["Times New Roman"]

    plt.style.use('bmh')
    
    fig = plt.figure(figsize=(10, 8), dpi=300)
    # fig.set_facecolor('None')
    text = fig.suptitle(fig_title)
    text.set_fontsize(20)

    for i in range(len(deep_learning)):
        if is_all_cup:
            ax = plt.subplot(111)
        else:
            if is_train:
                ax = plt.subplot(2, 3, i + 1)
            else:
                ax = plt.subplot(1, 2, i + 1)
            ax.set_title(data_filename[i])

        venn2(subsets=(
                        deep_learning[i] - overlaps[i], 
                        spectronaut[i] - overlaps[i],
                        overlaps[i]),
                        set_labels=['', ''],
                        normalize_to=1.0,
                        set_colors=['#BEB8DA', '#FA7F6F'],
                        alpha=0.6,
                        ax=ax
                        )
        
        venn2_circles(subsets=(
                                deep_learning[i] - overlaps[i], 
                                spectronaut[i] - overlaps[i],
                                overlaps[i]),
                                linestyle=(1, (5, 1, 1, 2)), # type: ignore
                                linewidth=2.0,
                                color='gray',
                                alpha=0.5,
                                ax=ax
        )
        ax.legend(['DeepLearning', 'Spectronaut'], loc='upper right')

    fig.tight_layout()
    fig.savefig(f"{save_rootpath + fig_title.split(' ')[-1]}.png", dpi=1000)

def add_item_to_dict(dict_name: Dict[str, Set[str]], key: str, item: str):
    dict_name[key].add(item)

def add_item_to_dict_List(dict_name: Dict[str, List], key: str, item: str):
    dict_name[key].append(item)

def calculate_cup(fileto_peptide_protein: Dict[str, Set[str]]):
    files = list(fileto_peptide_protein.keys())
    initial_file = files[0]
    cup = fileto_peptide_protein[initial_file].copy()
    for f in files[1:]:
        cup |= fileto_peptide_protein[f]
    return cup

def calculate_cap(fileto_peptide_protein: Dict[str, Set[str]]):
    files = list(fileto_peptide_protein.keys())
    initial_file = files[0]
    cap = fileto_peptide_protein[initial_file].copy()
    for i, f in enumerate(files[1:]):
        cap &= fileto_peptide_protein[f]
    return cap

def CCS_H15150_data(precursorId_file_label: Dict):
    CCS_filename_peptides: Dict[str, Set] = {}
    H15150_filename_peptides: Dict[str, Set]  = {}
    for key in precursorId_file_label.keys(): 
        file_name = key[1]
        if 'CCS' in file_name:
            CCS_filename_peptides[file_name] = set()
        else:
            H15150_filename_peptides[file_name] = set()

    for key in precursorId_file_label.keys():
        file_name = key[1]
        peptide = key[0]
        if 'CCS' in file_name:
            add_item_to_dict(CCS_filename_peptides, file_name, peptide)
        else:
            add_item_to_dict(H15150_filename_peptides, file_name, peptide)
    return CCS_filename_peptides, H15150_filename_peptides

def generate_modifiedPeptide_to_strippedPeptide_map(library: pd.DataFrame):
    modifiedPeptide_to_strippedPeptide_map = {}
    for modified_peptide, stripped_peptide in library[['ModifiedPeptide', 'StrippedPeptide']].values:
        modifiedPeptide_to_strippedPeptide_map[modified_peptide] = stripped_peptide
    return modifiedPeptide_to_strippedPeptide_map

def delete_file_result(d: Dict):
    """
    """
    wipe_d = {}
    for key, val in d.items():
        if len(val) < 200:
            continue
        wipe_d[key] = val
    return wipe_d

def generate_file_peptides(precursorId_file_label: Dict):
    filename_peptides: Dict[str, Set] = {key[1]:set() for key in precursorId_file_label.keys()}
    for key in precursorId_file_label.keys():
        file_name = key[1]
        peptide = key[0]
        add_item_to_dict(filename_peptides, file_name, peptide)
    return filename_peptides

def isin_strippedPeptide_to_modifiedPeptides_map(stripped_peptide: str, modified_peptide: str, strippedPeptide_to_modifiedPeptides_map: Dict[str, Set]):
    if modified_peptide in strippedPeptide_to_modifiedPeptides_map[stripped_peptide]:
        return True
    else:
        return False

def get_modifiedPeptideSet_first_element(strippedPeptide_to_modifiedPeptides_map: Dict[str, Set], stripped_peptide: str):
    for v in strippedPeptide_to_modifiedPeptides_map[stripped_peptide]:
        return v

def generate_sepctronaut_precursorId_file_label(df: pd.DataFrame, is_strippedPeptide: bool, strippedPeptide_to_modifiedPeptides_map: Dict[str, Set], file_names: List[str]):
    precursorId_label = {}
    # 前面的 column_name
    # ['PG.ProteinGroups', 'PG.ProteinAccessions', 'PG.Genes', 'PG.ProteinDescriptions', 'PEP.StrippedSequence', 'EG.PrecursorId']
    for row_data in df.values:
        for i, data in enumerate(row_data[6:]):
            if not np.isnan(data): # 把其中非 nan 的值筛选出来
                stripped_peptide, modifiedPeptide_charge = row_data[4], row_data[5]                 
                modified_peptide = str(modifiedPeptide_charge).split('.')[0]
                file_name = file_names[i].split('.d')[0].split(']')[1].strip()
                charge = int(str(modifiedPeptide_charge).split('.')[1])
                
                if is_strippedPeptide: 
                    key = stripped_peptide
                elif isin_strippedPeptide_to_modifiedPeptides_map(stripped_peptide, modified_peptide, strippedPeptide_to_modifiedPeptides_map):
                    key = (modified_peptide, charge)
                else:
                    modified_peptide = get_modifiedPeptideSet_first_element(strippedPeptide_to_modifiedPeptides_map, stripped_peptide)
                    key = (modified_peptide, charge)
                precursorId_label[(key, file_name)] = True
                
    return precursorId_label

def read_spectronaut_result(file_name: str, is_strippedPeptide: bool, strippedPeptide_to_modifiedPeptides_map: Dict[str, Set], isCSC_H15150: bool = False):
    df = pd.read_csv(file_name, sep='\t')
    file_names = df.columns[6:]
    precursorId_file_label = generate_sepctronaut_precursorId_file_label(df, is_strippedPeptide, strippedPeptide_to_modifiedPeptides_map, file_names) # type: ignore
    if isCSC_H15150:
        CCS_filename_peptides, H15150_filename_peptides = CCS_H15150_data(precursorId_file_label)
        return delete_file_result(CCS_filename_peptides), delete_file_result(H15150_filename_peptides)
    filename_peptide_set = generate_file_peptides(precursorId_file_label)
    return delete_file_result(filename_peptide_set)

def calculate_files_overlap(deeplearning: Dict, spectronaut: Dict):
    deeplearning_files = set([file for file in deeplearning.keys()])
    spectronaut_files = set([file for file in spectronaut.keys()])
    files = deeplearning_files & spectronaut_files
    return files

def calculate_deeplearning_spectronaut_overlap(deeplearning: Dict, spectronaut: Dict):    
    overlap = {}
    for file in deeplearning.keys():
        overlap[file] = deeplearning[file] & spectronaut[file]

    return overlap

def generate_deeplearning_precursorId_file_label(target_file: npt.NDArray, is_strippedPeptide: bool, modifiedPeptide_to_strippedPeptide_map: Dict):
    precursorId_label = {}
    for file_name, precursorId in target_file:
        if is_strippedPeptide:
            precursorId = modifiedPeptide_to_strippedPeptide_map[precursorId[0]]
        precursorId_label[(precursorId, file_name)] = True
    return precursorId_label

def fdrfilter_result(target_path: str, decoy_path: str):
    target_file, target = np.load(target_path, allow_pickle=True)
    _, decoy = np.load(decoy_path, allow_pickle=True)

    target_score = []
    decoy_score = []

    for score in target:
        target_score.extend(score)
    
    for score in decoy:
        decoy_score.extend(score)

    target_score = np.array(target_score)
    decoy_score = np.array(decoy_score)

    target_sort = np.sort(target_score)
    decoy_sort = np.sort(decoy_score)

    threshold = target_sort[0]

    # 计算界限值
    d_i = 0
    for i in tqdm(range(len(target_sort))):
        while decoy_sort[d_i] < target_sort[i]:
            d_i += 1
        fdr = (len(decoy_sort) - d_i) / (len(target_sort) - i)
        if fdr < 0.01:
            threshold = target_sort[i]
            break
    
    # 筛选出大于阈值的肽段及其所在文件
    return target_file[(target_score>=threshold).nonzero()]        

def read_deeplearning_result(target_path: str, decoy_path: str, is_strippedPeptide: bool, modifiedPeptide_to_strippedPeptide_map: Dict, isCSC_H15150: bool = False):
    fdrfilter_file = fdrfilter_result(target_path, decoy_path)
    precursorId_file_label = generate_deeplearning_precursorId_file_label(fdrfilter_file, is_strippedPeptide, modifiedPeptide_to_strippedPeptide_map)
    if isCSC_H15150:
        CCS_filename_peptides, H15150_filename_peptides = CCS_H15150_data(precursorId_file_label)
        return delete_file_result(CCS_filename_peptides), delete_file_result(H15150_filename_peptides)
    filename_peptide_set = generate_file_peptides(precursorId_file_label)
    return delete_file_result(filename_peptide_set)

def opt_for_files_overlap(previous_dict: Dict[str, Set[str]], files_overlap: Set[str]):
    new_dict = {}
    for file in files_overlap:
        new_dict[file] = previous_dict[file]
    return new_dict

def generate_sequence_to_protein_map(library: pd.DataFrame):
    sequence_to_protein = {}
    for sequence, protein_group in library[['StrippedPeptide', 'Protein Name']].values:
        if sequence not in sequence_to_protein.keys():
            sequence_to_protein[sequence] = protein_group
    return sequence_to_protein
        
def generate_modifiedPeptide_to_protein_map(library: pd.DataFrame):
    modifiedPeptide_to_protein = {}
    for modifiedPeptide, protein_group in library[['ModifiedPeptide', 'Protein Name']].values:
        if modifiedPeptide not in modifiedPeptide_to_protein.keys():
            modifiedPeptide_to_protein[modifiedPeptide] = protein_group
    return modifiedPeptide_to_protein

def generate_strippedPeptide_to_modifiedPeptideSet(library: pd.DataFrame):
    strippedPeptide_to_modifiedPeptides: Dict[str, Set] = {}
    for modified_peptide, stripped_peptide in library[['ModifiedPeptide', 'StrippedPeptide']].values:
        if stripped_peptide not in strippedPeptide_to_modifiedPeptides.keys():
            strippedPeptide_to_modifiedPeptides[stripped_peptide] = set()
        strippedPeptide_to_modifiedPeptides[stripped_peptide].add(modified_peptide)
    return strippedPeptide_to_modifiedPeptides

def calculate_proetin_result(protein_cluster: Dict[str, str]):
    pass

def find_proteins(peptide_set: Set[str], is_strippedPeptide, peptide_to_protein_map: Dict[str, str]):
    peptide_protein_list = []
    for p in peptide_set:
        peptide = p
        if not is_strippedPeptide:
            peptide = p[0]
        proteins = peptide_to_protein_map[peptide].split(';')
        for protein in proteins:
            peptide_protein_list.append((peptide, protein))
    # 去重, 防止添加重复的边
    return set(idpicker.find_valid_proteins(list(set(peptide_protein_list))).keys())

def calculate_average(d: Dict[str, Set[str]]):
    d_num = {key: len(value) for key, value in d.items()}
    average = int(sum(d_num.values()) / len(d_num.keys()))
    return average

def calculate_statistics_attributes(deeplearning_result: Dict[str, Set[str]], spectronaut_result: Dict[str, Set[str]]):
    """
    Return
    ---
    -   肽段/蛋白质平均数
    -   所有重复样本肽段/蛋白质并集
    -   所有重复样本肽段/蛋白质交集
    -   deeplearning-spectronaut overlap 的平均值
    """
    # 计算所有文件的并集
    spectronaut_result_cup = calculate_cup(spectronaut_result)
    deeplearning_result_cup = calculate_cup(deeplearning_result)

    spectronaut_result_cap = calculate_cap(spectronaut_result)
    deeplearning_result_cap = calculate_cap(deeplearning_result)

    # 计算各个文件的 overlap
    spectronaut_deeplearning_overlap = calculate_deeplearning_spectronaut_overlap(deeplearning_result, spectronaut_result)

    # 计算各项平均值
    # 1. 每个文件检测到的肽段数均值
    spectronaut_result_average = calculate_average(spectronaut_result)
    deeplearning_result_average = calculate_average(deeplearning_result)

    # 2. overlap 的均值
    spectronaut_deeplearning_overlap_average = calculate_average(spectronaut_deeplearning_overlap)

    average = {
        'spectronaut': spectronaut_result_average,
        'deeplearning': deeplearning_result_average
    }

    cup = {
        'spectronaut': spectronaut_result_cup,
        'deeplearning': deeplearning_result_cup
    }

    cap = {
        'spectronaut': spectronaut_result_cap,
        'deeplearning': deeplearning_result_cap
    }

    overlap_average = spectronaut_deeplearning_overlap_average

    return average, cup, cap, overlap_average
    
def print_dataset_result(average:Dict[str, int], cup: Dict[str, Set[str]], cap: Dict[str, Set[str]], overlap_average: int, is_peptide: bool = True):
    if is_peptide:
        print('#' * 5 + ' peptide_result ' + '#'* 5)
    else:
        print('#' * 5 + ' protein_result ' + '#'* 5)

    print('\t' + 'spectronaut_cup_num: ', len(cup['spectronaut']), f"spectronaut_average: {average['spectronaut']}", 'spectronaut_cap_num: ', len(cap['spectronaut']))
    print('\t' + 'deeplearning_cup_num: ', len(cup['deeplearning']), f"deeplearning_average: {average['deeplearning']}", 'deeplearning_cap_num: ', len(cap['deeplearning']))
    print('\t' + 'overlap: ', f'deeplearning_spectronaut_average: {overlap_average}')
    print('\t' + 'cap_overlap: 'f'{len(cap["deeplearning"] & cap["spectronaut"])}')

def print_all_dataset_result(deeplearning_all_cup, spectronaut_all_cup, is_peptide: bool = True):
    if is_peptide:
        print('#' * 5 + ' peptide_result ' + '#'* 5)
    else:
        print('#' * 5 + ' protein_result ' + '#'* 5)

    print('\t' + f'deeplearning: {len(deeplearning_all_cup)}')
    print('\t' + f'spectronaut: {len(spectronaut_all_cup)}')
    print('\t' + f'overlap: {len(deeplearning_all_cup & spectronaut_all_cup)}')

def calculate_peptide_protein_overlap(deeplearning_peptide_result: Dict[str, Set], spectronaut_peptide_result: Dict[str, Set], is_strippedPeptide: bool, peptide_protein_map: Dict[str, str]):
    # 筛选出公有的文件
    files_overlap = calculate_files_overlap(deeplearning_peptide_result, spectronaut_peptide_result)
    print(len(files_overlap))
    spectronaut_peptide_result = opt_for_files_overlap(spectronaut_peptide_result, files_overlap)
    deeplearning_peptide_result = opt_for_files_overlap(deeplearning_peptide_result, files_overlap)
    # 得到每个文件下检测的蛋白质结果
    spectronaut_protein_result = {
        file: find_proteins(peptide_set, is_strippedPeptide, peptide_protein_map) 
        for file, peptide_set in spectronaut_peptide_result.items()
    }
    deeplearning_protein_result = {
        file: find_proteins(peptide_set, is_strippedPeptide, peptide_protein_map) 
        for file, peptide_set in deeplearning_peptide_result.items()
    }

    protein_len = list({key: len(value) for key, value in spectronaut_protein_result.items()}.values())

    # 肽段的统计结果
    peptide_average, peptide_cup, peptide_cap, peptide_overlap_average = calculate_statistics_attributes(deeplearning_peptide_result, spectronaut_peptide_result)
    print_dataset_result(peptide_average, peptide_cup, peptide_cap, peptide_overlap_average)
    
    protein_average, protein_cup, protein_cap, protein_overlap_average = calculate_statistics_attributes(deeplearning_protein_result, spectronaut_protein_result)
    print_dataset_result(protein_average, protein_cup, protein_cap, protein_overlap_average, False)

    final_result = {
        'peptide': {
            'average': peptide_average,
            'cup': peptide_cup,
            'cap': peptide_cap,
            'overlap_average': peptide_overlap_average
        },
        'protein': {
            'average': protein_average,
            'cup': protein_cup,
            'cap': protein_cap,
            'overlap_average': protein_overlap_average,
            'protein_len': protein_len
        }
    }
    return final_result

def get_fig_title(is_train, is_all_dataset):
    if is_train:
        if is_all_dataset:
            return 'all_train_dataset_cup'
        else:
            return 'train_dataset_result'
    else:
        if is_all_dataset:
            return 'all_test_dataset_cup'
        else:
            return 'test_dataset_result'

def get_save_rootpath(is_peptide, is_cap=False):
    if is_peptide:
        if is_cap:
            return "./count_result/peptide/cap/"
        return "./count_result/peptide/"
    else:
        if is_cap:
            return "./count_result/protein/cap/"
        return "./count_result/protein/"

def initial_params():
    spectronaut_all_cup = {
        'peptide': set(),
        'protein': set()
    }

    deeplearning_all_cup = {
        'peptide': set(),
        'protein': set()
    }

    spectronaut_average = {
        'peptide': [],
        'protein': []
    }

    deeplearning_average = {
        'peptide': [],
        'protein': []
    }

    spectronaut_deeplearning_overlap_average = {
        'peptide': [],
        'protein': []
    }

    spectronaut_cap = {
        'peptide': [],
        'protein': []
    }

    deeplearning_cap = {
        'peptide': [],
        'protein': []
    }

    spectronaut_deeplearning_cap_overlap = {
        'peptide': [],
        'protein': []
    }

    files = []
    return spectronaut_all_cup, deeplearning_all_cup, spectronaut_average, deeplearning_average, spectronaut_deeplearning_overlap_average, spectronaut_cap, deeplearning_cap, spectronaut_deeplearning_cap_overlap, files

def generate_result(is_strippedPeptide: bool = True, is_train: bool = True):
    root = './train_result/'
    if not is_train:
        root = './test_result/'
    if is_train:
        print('\t' * 5 + '#' * 10 + ' TrainDataResult ' + '#' * 10)
    else:
        print('\t' * 5 + '#' * 10 + ' TestDataResult ' + '#' * 10)

    library = pd.read_csv('./library/AD8-300S-directDIA.xls', sep='\t')
    strippedPeptide_to_modifiedPeptides_map = generate_strippedPeptide_to_modifiedPeptideSet(library)
    modifiedPeptide_to_strippedPeptide_map = generate_modifiedPeptide_to_strippedPeptide_map(library)
    sequence_to_protein_map = generate_sequence_to_protein_map(library)
    modifiedPeptide_to_protein_map = generate_modifiedPeptide_to_protein_map(library)

    to_protein_map = modifiedPeptide_to_protein_map
    if is_strippedPeptide:
        to_protein_map = sequence_to_protein_map

    spectronaut_all_cup, deeplearning_all_cup, spectronaut_average, deeplearning_average, spectronaut_deeplearning_overlap_average, spectronaut_cap, deeplearning_cap, spectronaut_deeplearning_cap_overlap, files = initial_params()

    # flag = True

    for dataset_dir in os.listdir(root):
        dataset_name = dataset_dir.split('_')[0]

        print('#' * 5 + f" {dataset_name}_result " + '#' * 5)

        dataset_dir = root + dataset_dir + '/'
        
        spectronaut_filepath = [dataset_dir + file for file in os.listdir(dataset_dir) if file.endswith('.tsv')][0]
        
        deeplearning_target_filepath = [dataset_dir + file for file in os.listdir(dataset_dir) if 'target' in file][0]
        deeplearning_decoy_filepath = [dataset_dir + file for file in os.listdir(dataset_dir) if 'decoy' in file][0]

        is_dataset5 = 'dataset5' in spectronaut_filepath
        is_dataset6 = 'dataset6' in spectronaut_filepath
        
        if is_dataset5 or is_dataset6:
            # 'CSS' 是 dataset5, 'H15150' 是dataset6
            spectronaut_result_CSS, spectronaut_result_H15150 = read_spectronaut_result(spectronaut_filepath, is_strippedPeptide, strippedPeptide_to_modifiedPeptides_map, True)
            deeplearning_result_CSS, deeplearning_result_H15150 = read_deeplearning_result(deeplearning_target_filepath, deeplearning_decoy_filepath, is_strippedPeptide, modifiedPeptide_to_strippedPeptide_map, True)

            if is_dataset5:
                final_result = calculate_peptide_protein_overlap(deeplearning_result_CSS, spectronaut_result_CSS, is_strippedPeptide, to_protein_map) # type: ignore
            else:
                final_result = calculate_peptide_protein_overlap(deeplearning_result_H15150, spectronaut_result_H15150, is_strippedPeptide, to_protein_map) # type: ignore
        else:
            # 初始结果
            spectronaut_result = read_spectronaut_result(spectronaut_filepath, is_strippedPeptide, strippedPeptide_to_modifiedPeptides_map)
            deeplearning_result = read_deeplearning_result(deeplearning_target_filepath, deeplearning_decoy_filepath, is_strippedPeptide, modifiedPeptide_to_strippedPeptide_map)
            final_result = calculate_peptide_protein_overlap(deeplearning_result, spectronaut_result, is_strippedPeptide, to_protein_map) # type: ignore
        """
        final_result = {
            'peptide': {
                'average': peptide_average,
                'cup': peptide_cup,
                'overlap_average': peptide_overlap_average
            },
            'protein': {
                'average': protein_average,
                'cup': protein_cup,
                'overlap_average': protein_overlap_average
            }
        }
        """
        spectronaut_all_cup['peptide'] = spectronaut_all_cup['peptide'] | final_result['peptide']['cup']['spectronaut']
        deeplearning_all_cup['peptide'] = deeplearning_all_cup['peptide'] | final_result['peptide']['cup']['deeplearning']
        spectronaut_all_cup['protein'] = spectronaut_all_cup['protein'] | final_result['protein']['cup']['spectronaut']
        deeplearning_all_cup['protein'] = deeplearning_all_cup['protein'] | final_result['protein']['cup']['deeplearning']

        spectronaut_average['peptide'].append(final_result['peptide']['average']['spectronaut'])
        deeplearning_average['peptide'].append(final_result['peptide']['average']['deeplearning'])
        spectronaut_average['protein'].append(final_result['protein']['average']['spectronaut'])
        deeplearning_average['protein'].append(final_result['protein']['average']['deeplearning'])

        spectronaut_cap['peptide'].append(len(final_result['peptide']['cap']['spectronaut']))
        deeplearning_cap['peptide'].append(len(final_result['peptide']['cap']['deeplearning']))
        spectronaut_cap['protein'].append(len(final_result['protein']['cap']['spectronaut']))
        deeplearning_cap['protein'].append(len(final_result['protein']['cap']['deeplearning']))

        spectronaut_deeplearning_cap_overlap['peptide'].append(len(final_result['peptide']['cap']['spectronaut'] & final_result['peptide']['cap']['deeplearning']))
        spectronaut_deeplearning_cap_overlap['protein'].append(len(final_result['protein']['cap']['spectronaut'] & final_result['protein']['cap']['deeplearning']))

        spectronaut_deeplearning_overlap_average['peptide'].append(final_result['peptide']['overlap_average'])
        spectronaut_deeplearning_overlap_average['protein'].append(final_result['protein']['overlap_average'])
        files.append(dataset_name)

    if is_train:
        print(f'All train_dataset: ')
    else:
        print(f'All test_dataset: ')

    fig_title = get_fig_title(is_train, False)
    fig_title = 'Peptide: ' + fig_title 
    save_rootpath = get_save_rootpath(True)
    # print_all_dataset_result(deeplearning_all_cup['peptide'], spectronaut_all_cup['peptide'])
    plot_venn(deeplearning_average['peptide'], spectronaut_average['peptide'], spectronaut_deeplearning_overlap_average['peptide'], files, is_train, fig_title, save_rootpath)
    
    save_rootpath = get_save_rootpath(True, True)
    plot_venn(deeplearning_cap['peptide'], spectronaut_cap['peptide'], spectronaut_deeplearning_cap_overlap['peptide'], files, is_train, fig_title, save_rootpath)


    fig_title = get_fig_title(is_train, False)
    fig_title = 'Protein: ' + fig_title
    save_rootpath = get_save_rootpath(False)
    # print_all_dataset_result(deeplearning_all_cup['protein'], spectronaut_all_cup['protein'], False)
    plot_venn(deeplearning_average['protein'], spectronaut_average['protein'], spectronaut_deeplearning_overlap_average['protein'], files, is_train, fig_title, save_rootpath)

    save_rootpath = get_save_rootpath(False, True)
    plot_venn(deeplearning_cap['protein'], spectronaut_cap['protein'], spectronaut_deeplearning_cap_overlap['protein'], files, is_train, fig_title, save_rootpath)

    fig_title = get_fig_title(is_train, True)
    fig_title = 'Peptide: ' + fig_title
    save_rootpath = get_save_rootpath(True)
    plot_venn(
            [len(deeplearning_all_cup['peptide'])], 
            [len(spectronaut_all_cup['peptide'])], 
            [len(deeplearning_all_cup['peptide'] & spectronaut_all_cup['peptide'])], 
            [],
            is_train,
            fig_title,
            save_rootpath,
            True
)
    
    fig_title = get_fig_title(is_train, True)
    fig_title = 'Protein: ' + fig_title
    save_rootpath = get_save_rootpath(False)
    plot_venn(
            [len(deeplearning_all_cup['protein'])], 
            [len(spectronaut_all_cup['protein'])], 
            [len(deeplearning_all_cup['protein'] & spectronaut_all_cup['protein'])], 
            [],
            is_train,
            fig_title,
            save_rootpath,
            True
)

if __name__ == "__main__":
    is_strippedPeptides = [True, True]
    is_trains = [True, False]
    for i in range(len(is_strippedPeptides)):
        generate_result(is_strippedPeptides[i], is_trains[i])
