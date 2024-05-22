from typing import Dict, Any, Literal
import time

import torch
import numpy as np
import numpy.typing as npt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm


def cal_tp_fp_fn(y: torch.Tensor, y_pred: torch.Tensor):
    tp_fp = torch.sum(y_pred == 1).item()
    tp_fn = torch.sum(y == 1).item()
    tp = torch.sum((y_pred == 1) & (y == 1)).item()
    return {
        'TP': tp,
        'TP+FP': tp_fp,
        'TP+FN': tp_fn
    }


def is_replace(best_model_info, val_info, select_feature: Literal['loss', 'relative_loss', 'accuracy', 'f1_score']):
    if select_feature == 'loss' or select_feature == 'relative_loss':
        if best_model_info[select_feature] > val_info[select_feature]:
            return True

    elif select_feature == 'accuracy' or select_feature == 'f1_score':
        if best_model_info[select_feature] < val_info[select_feature]:
            return True

    return False


def classfier_train(
<<<<<<< HEAD
    model: Module,
    device: torch.device,
    optim: Optimizer,
    loss_func: Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    configs: Dict[str, Any],
    train_params_info_save: Dict[str, Any],
    select_feature: Literal['loss', 'accuracy', 'f1_score']
=======
        model: Module,
        device: torch.device,
        optim: Optimizer,
        loss_func: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        configs: Dict[str, Any],
        train_params_info_save: Dict[str, Any],
        select_feature: Literal['loss', 'accuracy', 'f1_score']
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
):
    """
        模型训练通用函数

        Parameters:
        ---
        -   model: 待训练的模型
        -   device: 设备
        -   optim: 优化器
        -   loss_func: 损失计算函数
        -   train_dataloader: 训练数据的 dataloader
        -   val_dataloader: 验证集数据的 dataloader
        -   configs: 配置信息
        -   train_params_info_save: 保存训练时指标

            key:
                'train_loss'
                'val_loss'
                'train_accuracy'
                'val_accuracy'
                'train_f1_score'
                'val_f1_score'
    """
    model.train()
    best_model_info = {
        'loss': torch.inf,
        'accuracy': -torch.inf,
        'f1_score': -torch.inf,
    }
    best_epoch = 0
    end = 0
    best_model = None
    for epoch in range(configs['n_epoch']):
        if end == configs['early_end']:
            break
        info = {
            'correct': 0,
            'loss': 0,
            'length': 0,
            'TP': 0,
            'TP+FP': 0,
            'TP+FN': 0
        }
        # for batch in tqdm(train_dataloader, f'epoch: {epoch + 1}'):
<<<<<<< HEAD
        for batch, _ in train_dataloader:
=======
        for batch in train_dataloader:
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
            y = batch['label'].to(device)
            x = {
                feature: batch[feature].to(device)
                for feature in configs['features']['use']
            }

            y_pred = model(**x)
            y_pred = torch.squeeze(y_pred, dim=1)
            loss = loss_func(y_pred, y)

            preds = torch.zeros_like(y_pred, device=device)
            preds[y_pred <= configs['score_threshold']] = 0
            preds[y_pred > configs['score_threshold']] = 1

            # loss, accuracy
            info['correct'] += torch.sum(preds == y).item()  # type: ignore
            info['length'] += y.shape[0]
            info['loss'] += loss.item() * y.shape[0]
            # f1 score
            f1 = cal_tp_fp_fn(y, preds)
            info['TP'] += f1['TP']  # type: ignore
            info['TP+FP'] += f1['TP+FP']  # type: ignore
            info['TP+FN'] += f1['TP+FN']  # type: ignore
            optim.zero_grad()
            loss.backward()
            optim.step()
        # end batch
        train_loss = info['loss'] / info['length']
        train_accuracy = info['correct'] / info['length']

        train_params_info_save['train_loss'].append(train_loss)
        train_params_info_save['train_accuracy'].append(train_accuracy)

        # f1 score 计算公式
        precision = info['TP'] / info['TP+FP']
        recall = info['TP'] / info['TP+FN']
        train_f1_score = 2 * precision * recall / (precision + recall)
        train_params_info_save['train_f1_score'].append(train_f1_score)

        print(
            f"epoch {epoch + 1}, train info. (accuracy={train_accuracy:.4f}, loss={train_loss:.4f}, f1_score={train_f1_score:.4f})")

        # 通过验证集筛选模型
        val_info = classfier_evaluate(
            model=model,
            device=device,
            val_dataloader=val_dataloader,
            configs=configs,
            loss_func=loss_func
        )
        train_params_info_save['val_loss'].append(val_info['loss'])
        train_params_info_save['val_accuracy'].append(val_info['accuracy'])
        train_params_info_save['val_f1_score'].append(val_info['f1_score'])

        if is_replace(best_model_info, val_info, select_feature):
            for key in best_model_info.keys():
                best_model_info[key] = val_info[key]
            best_model = model.state_dict()
            best_epoch = epoch
            print(
                f"epoch {epoch + 1}, best model saved. (accuracy={best_model_info['accuracy']:.4f}, loss={best_model_info['loss']:.4f}, f1_score={best_model_info['f1_score']:.4f})")
            end = 0
        else:
            end += 1
    # end epoch
    torch.save(best_model, configs['model_save_path'])
    print(
        f"epoch {best_epoch + 1}, best model saved. (accuracy={best_model_info['accuracy']:.4f}, loss={best_model_info['loss']:.4f}, f1_score={best_model_info['f1_score']:.4f})")


def classfier_evaluate(
<<<<<<< HEAD
    model: Module,
    device: torch.device,
    val_dataloader: DataLoader,
    configs: Dict[str, Any],
    loss_func: Module,
=======
        model: Module,
        device: torch.device,
        val_dataloader: DataLoader,
        configs: Dict[str, Any],
        loss_func: Module,
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
):
    model.eval()
    info = {
        'correct': 0,
        'length': 0,
        'loss': 0,
        'TP': 0,
        'TP+FP': 0,
        'TP+FN': 0
    }
    # for batch in tqdm(val_dataloader, 'validation: '):
<<<<<<< HEAD
    for batch, _ in val_dataloader:
=======
    for batch in val_dataloader:
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
        y = batch['label'].to(device)
        x = {
            feature: batch[feature].to(device)
            for feature in configs['features']['use']
        }

        y_pred = model(**x)
        y_pred = torch.squeeze(y_pred, dim=1)
        loss = loss_func(y_pred, y)

        preds = torch.zeros_like(y_pred, device=device)
        preds[y_pred <= configs['score_threshold']] = 0
        preds[y_pred > configs['score_threshold']] = 1

        info['correct'] += torch.sum(preds == y).item()  # type: ignore
        info['length'] += y.shape[0]
        info['loss'] += loss.item() * y.shape[0]

        f1 = cal_tp_fp_fn(y, preds)

        info['TP'] += f1['TP']  # type: ignore
        info['TP+FN'] += f1['TP+FN']  # type: ignore
        info['TP+FP'] += f1['TP+FP']  # type: ignore

    precision, recall = info['TP'] / info['TP+FP'], info['TP'] / info['TP+FN']
    return {
        'loss': info['loss'] / info['length'],
        'accuracy': info['correct'] / info['length'],
        'f1_score': 2 * precision * recall / (precision + recall)
        # 可以添加其他信息, 例如 roc 曲线面积等等
    }


def classfier_test(
<<<<<<< HEAD
    model: Module,
    device: torch.device,
    dataloader: DataLoader,
    configs: Dict[str, Any]
):
    model.eval()
    score = []
    infos = []
    with torch.no_grad():
        # for batch in tqdm(dataloader, 'test'):
        for batch, info in dataloader:
=======
        model: Module,
        device: torch.device,
        dataloader: DataLoader,
        info: npt.NDArray,
        configs: Dict[str, Any]
):
    model.eval()
    score = []
    with torch.no_grad():
        # for batch in tqdm(dataloader, 'test'):
        for batch in dataloader:
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
            x = {
                feature: batch[feature].to(device)
                for feature in configs['features']['use']
            }

            y_pred: torch.Tensor = model(**x)
            y_pred = torch.squeeze(y_pred, dim=1)
            score.append(y_pred.cpu().numpy())
<<<<<<< HEAD
            infos.append(info)

    return {
        'score': np.concatenate(score, axis=0),
        'info': np.concatenate(infos, axis=0)
=======

    return {
        'score': np.concatenate(score, axis=0),
        'info': info
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
    }


def regression_train(
    model: Module,
    device: torch.device,
    optim: Optimizer,
    loss_func: Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    configs: Dict[str, Any],
    train_params_info_save: Dict[str, Any],
    select_feature: Literal['loss', 'relative_loss']
):
    """
        Trains a regression model.

        Parameters:
        - model: The model to be trained, derived from the Module class.
        - device: Specifies the device for the training and validation data loaders to operate on.
        - optim: The optimizer instance used to update model parameters.
        - loss_func: The defined loss function module.
        - train_dataloader: DataLoader for training data.
        - val_dataloader: DataLoader for validation data.
        - configs: A dictionary containing various configurations such as learning rate.
        - train_params_info_save: A dictionary to store training parameters and information.
        - select_feature: The feature selection method, either 'loss' or 'relative_loss'.

        Returns:
        None
    """
    model.train()
    best_model_info = {
        'loss': torch.inf,
        'relative_loss': torch.inf,
    }
    best_epoch = 0
    end = 0
    best_model = None
    for epoch in range(configs['n_epoch']):
        if end == configs['early_end']:
            break
        info = {
            'loss': 0,
            'relative_loss': 0,
            'length': 0
        }
        # for batch, _ in tqdm(train_dataloader, f'epoch: {epoch+1}'):
        for batch, _ in train_dataloader:
            optim.zero_grad()
            # y: torch.Tensor = torch.log2(batch['label'].to(device))
            y: torch.Tensor = batch['label'].to(device)
            x = {
                feature: batch[feature].to(device)
                for feature in configs['features']['use']
            }

            y_pred = model(**x)
            y_pred = torch.squeeze(y_pred, dim=1)
            loss: torch.Tensor = loss_func(y_pred, y)
            # y = torch.add(y, 1e-4)
            mask = y > 1
            relative_loss = torch.sum(
                torch.abs(y_pred[mask] - y[mask]) / y[mask])
            info['relative_loss'] += relative_loss.item()
            info['loss'] += loss.item() * y.shape[0]
            info['length'] += y.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(
                model.parameters(), 0.5)
            optim.step()
            # torch.cuda.empty_cache()  # 清理显存
        # end batch
        train_loss = info['loss'] / info['length']
        train_relative_loss = info['relative_loss'] / info['length']
        train_params_info_save['train_loss'].append(train_loss)
        train_params_info_save['train_relative_loss'].append(
            train_relative_loss)

        print(
            f"epoch {epoch + 1}, train info. (loss={train_loss:.4f}, relative_loss={train_relative_loss:.4f})")
        # torch.cuda.empty_cache()  # 清理显存
        # 通过验证集筛选模型
        val_info = regression_evaluate(
            model=model,
            device=device,
            val_dataloader=val_dataloader,
            configs=configs,
            loss_func=loss_func
        )
        train_params_info_save['val_loss'].append(val_info['loss'])
        train_params_info_save['val_relative_loss'].append(
            val_info['relative_loss'])
        if is_replace(best_model_info, val_info, select_feature):
            for key in best_model_info.keys():
                best_model_info[key] = val_info[key]
            best_model = model.state_dict()
            best_epoch = epoch
            print(
                f"epoch {epoch + 1}, best model saved. (loss={best_model_info['loss']:.4f}, relative_loss={best_model_info['relative_loss']:.4f})")
            end = 0
        else:
            end += 1
    # end epoch
    torch.save(best_model, configs['model_save_path'])
    print(
        f"epoch {best_epoch + 1}, best model saved. (loss={best_model_info['loss']:.4f}, relative_loss={best_model_info['relative_loss']:.4f})")


def regression_evaluate(
    model: Module,
    device: torch.device,
    val_dataloader: DataLoader,
    configs: Dict[str, Any],
    loss_func: Module,
<<<<<<< HEAD
=======


>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
):
    model.eval()
    info = {
        'loss': 0,
        'relative_loss': 0,
        'length': 0
    }
    with torch.no_grad():
        # for batch, _ in tqdm(val_dataloader, 'validation: '):
        for batch, _ in val_dataloader:
            # y: torch.Tensor = torch.log2(batch['label'].to(device))
            y: torch.Tensor = batch['label'].to(device)
            x = {
                feature: batch[feature].to(device)
                for feature in configs['features']['use']
            }

            y_pred = model(**x)
            y_pred = torch.squeeze(y_pred, dim=1)
            loss = loss_func(y_pred, y)
            mask = y > 1
            relative_loss = torch.sum(
                torch.abs(y_pred[mask] - y[mask]) / y[mask])
            info['length'] += y.shape[0]
            info['loss'] += loss.item() * y.shape[0]
            info['relative_loss'] += relative_loss.item()
            # torch.cuda.empty_cache()  # 清理显存

    return {
        'loss': info['loss'] / info['length'],
        'relative_loss': info['relative_loss'] / info['length']
    }


def regression_test(
    model: Module,
    device: torch.device,
    dataloader: DataLoader,
    configs: Dict[str, Any]
):
    model.eval()
    quantity = []
    infos = []
    with torch.no_grad():
        # for batch, info in tqdm(dataloader, 'test'):
        for batch, info in dataloader:
            x = {
                feature: batch[feature].to(device)
                for feature in configs['features']['use']
            }

            y_pred: torch.Tensor = model(**x)
            y_pred = torch.squeeze(y_pred, dim=1)
<<<<<<< HEAD
            quantity.append(y_pred.cpu().numpy())
            infos.append(info)
=======
            # print(y_pred)
            # y_pred = torch.pow(2, y_pred)
            # print(y_pred)
            quantity.append(y_pred.cpu().numpy())
            infos.append(info)
            # time.sleep(0.2)
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79

    return {
        'quantity': np.concatenate(quantity, axis=0),
        'info': np.concatenate(infos, axis=0)
    }
