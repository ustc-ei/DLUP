from typing import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from generatedata.utils.utils import read_dict_npy


def gen_df(data, columns, rename_map, dataset_type):
    df = pd.DataFrame(
        data, columns=columns
    )
    df['epoch'] = range(1, len(df) + 1)
    df['dataset_type'] = dataset_type
    df = df.rename(columns=rename_map)
    return df


def data_to_df(data):
    train_df = gen_df(
        data=data,
        columns=['train_loss', 'train_accuracy', 'train_f1_score'],
        rename_map={
            'train_loss': 'loss',
            'train_accuracy': 'accuracy',
            'train_f1_score': 'f1_score'
        },
        dataset_type='train'
    )

    val_df = gen_df(
        data=data,
        columns=['val_loss', 'val_accuracy', 'val_f1_score'],
        rename_map={
            'val_loss': 'loss',
            'val_accuracy': 'accuracy',
            'val_f1_score': 'f1_score'
        },
        dataset_type='validation'
    )

    return pd.concat((train_df, val_df))


def initila_ax(ax: Axes, title: str):
    ax.set_ylabel(' ')
    ax.legend().remove()
    ax.set_title(title)


def main(metrics_path: str):
    sns.set_style('whitegrid')

    colors = sns.color_palette(palette='Set2', n_colors=2)

    axs: Sequence[Sequence[Axes]]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), dpi=500)

    d = read_dict_npy(metrics_path)
    df = data_to_df(d)

    titles = ['loss', 'accuracy', 'f1_score']

    sns.lineplot(data=df, x='epoch', y='loss',
                 palette=colors, hue='dataset_type', ax=axs[0][0])

    handles, labels = axs[0][0].get_legend_handles_labels()
    initila_ax(axs[0][0], titles[0])

    sns.lineplot(data=df, x='epoch', y='accuracy',
                 palette=colors, hue='dataset_type', ax=axs[0][1])
    initila_ax(axs[0][1], titles[1])

    sns.lineplot(data=df, x='epoch', y='f1_score',
                 palette=colors, hue='dataset_type', ax=axs[1][0])
    initila_ax(axs[1][0], titles[2])

    fig.delaxes(axs[1][1])

    fig.legend(handles=handles, labels=labels, loc='lower left',
               bbox_to_anchor=(0.6, 0.2), ncol=1, frameon=False, prop={'size': 20})
    fig.tight_layout()

    fig.savefig('figs/train_metrics.png',
                bbox_inches='tight', facecolor='none')
