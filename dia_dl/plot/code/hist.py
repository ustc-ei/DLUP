import os
from typing import Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from generatedata.utils.utils import read_json, read_label_result_path, read_dict_npy


def data_to_df(target, decoy):
    target_df = pd.DataFrame(target, columns=['score'])
    target_df['type'] = 'target'
    decoy_df = pd.DataFrame(decoy, columns=['score'])
    decoy_df['type'] = 'decoy'
    return pd.concat([target_df, decoy_df])


def main(configs):
    data_name_sequence = [
        'train',
        'test',
    ]
    for data_name in data_name_sequence:
        result_target_path = os.path.join('./result', data_name, 'target.npy')

        result_decoy_path = os.path.join('./result', data_name, 'decoy.npy')

        target = read_dict_npy(result_target_path)
        decoy = read_dict_npy(result_decoy_path)

        df = data_to_df(target['score'], decoy['score'])
        axs: Sequence[Axes]
        sns.set_style('whitegrid')
        plt.subplot
        fig, axs = plt.subplots(figsize=(6, 3), dpi=500, nrows=1, ncols=2)
        colors = sns.color_palette('Set1')
        sns.set_palette(colors, 2)
        sns.histplot(data=df, x='score', hue='type', bins=50, ax=axs[0])
        sns.kdeplot(data=df, x='score', hue='type', ax=axs[1])
        sns.move_legend(axs[0], ncol=2, loc='lower left',
                        bbox_to_anchor=(0, 1.05), title=None, frameon=False)
        sns.move_legend(axs[1], ncol=2, loc='lower left',
                        bbox_to_anchor=(0, 1.05), title=None, frameon=False)
        fig.tight_layout()
        fig.savefig(f'figs/{data_name}_score{configs["extend"]}.png')


if __name__ == "__main__":
    main()
