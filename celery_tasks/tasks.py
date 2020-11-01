# coding: utf-8

import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from celery import Celery, Task
from celery.utils.log import get_task_logger
from pycm import ConfusionMatrix

import lwmw

path = os.path.dirname(__file__)
upload_path = os.path.join(path, '..', 'uploads')

logger = get_task_logger(__name__)


def pixel_accuracy(predict, true_mask):
    """Метрика попиксильной точности
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение метрики попиксильной точности
    """
    return np.sum(predict * true_mask) / (np.sum(true_mask))


def Jaccard(predict, true_mask):
    """Мера Жаккара
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Жаккара точности
    """
    return sum(predict * true_mask) / (sum(predict) + sum(true_mask) - sum(predict * true_mask))


def Sorensen(predict, true_mask):
    """Мера Соренсена
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Соренсена точности
    """
    return (2 * sum(predict * true_mask)) / (sum(predict) + sum(true_mask))


def Kulchinski(predict, true_mask):
    """Мера Кульчински
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Кульчински точности
    """
    return ((sum(predict * true_mask)) / 2) * (1 / sum(predict) + 1 / sum(true_mask))


def Simpson(predict, true_mask):
    """Мера Симпсона
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Симпсона точности
    """
    return (2 * sum(predict * true_mask)) / (sum(predict) + sum(true_mask) - abs(sum(predict) - sum(true_mask)))


def Braun(predict, true_mask):
    """Мера Брауна
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Брауна точности
    """
    return (2 * sum(predict * true_mask)) / (sum(predict) + sum(true_mask) + abs(sum(predict) - sum(true_mask)))


class CompareMarkups(Task):
    name = 'compare_markups'
    # throws = (ValueError,)

    def get_metrics_table(self, doc_markup, ai_markup, norm=255.0):
        """Получение таблицы (pandas.DataFrame) метрик на основе сегментированных данных ИНС и эксперта
        Arguments:
            doc_markup (str): путь до PNG, размеченного ИНС
            ai_markup (str): путь до PNG, размеченного экспертом
            norm (float): значение нормировки изображений (Изображение / norm), default: 255.0
        Returns:
            metrics (pandas.DataFrame): таблица со значениями рассчитаных метрик
        """
        img_pred_data = [cv2.imread(ai_markup) / norm]
        img_true_data = [cv2.imread(doc_markup) / norm]

        metrics = pd.DataFrame()
        for i in range(1):
            temp_overal = ConfusionMatrix(actual_vector=img_true_data[i].ravel(), predict_vector=img_pred_data[i].ravel())
            temp_metrics = {
                "name": doc_markup,
                "PA": pixel_accuracy(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                "Jaccard": Jaccard(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                "Sorensen": Sorensen(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                "Kulchinski": Kulchinski(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                "Simpson": Simpson(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                "Braun_Blanke": Braun(img_pred_data[i].ravel(), img_true_data[i].ravel())
            }

            temp_overal = temp_overal.overall_stat
            for j in temp_overal.keys():
                temp_metrics[j] = temp_overal[j]

            metrics = metrics.append(temp_metrics, ignore_index=True)

        metrics = metrics[[
            "name",
            "ACC Macro",
            "Bangdiwala B",
            "Bennett S",
            "Conditional Entropy",
            "Cross Entropy",
            "F1 Micro",
            "FNR Micro",
            "FPR Micro",
            "Gwet AC1",
            "Hamming Loss",
            "Joint Entropy",
            "Kappa No Prevalence",
            "Mutual Information",
            "NIR",
            "Overall ACC",
            "Overall RACC",
            "Overall RACCU",
            "PPV Micro",
            "Reference Entropy",
            "Response Entropy",
            "Standard Error",
            "TNR Micro",
            "TPR Micro"
        ]]
        metrics = metrics.set_index("name")
        return metrics

    def predict(self, metrics, path_to_weights):
        """Оценка разметки ИНС
        Arguments:
            metrics (pandas.DataFrame): таблица метрик (из метода _get_metrics_table())
            path_to_weights (str): путь до файла с весами персептрона
            device (str): используемое устройство для вычисления default: "/gpu:1"
        Returns:
            metrics (pandas.Series): ряд с предсказаниями оценки эксперта
            grads (dict): данные для отрисовки графика значимости мер
        """
        settings = {
            "outs": 5,
            "input_len": len(metrics),
            "architecture": [31, 18],
            "inputs": len(metrics.columns),
            "activation": "sigmoid"
        }
        p = np.load(path_to_weights)

        predicts, grads = lwmw.predict(p, settings, metrics.values)

        for i in range(0, settings["outs"]):
            metrics["preds_" + str(i + 1)] = predicts[:, i]
        metrics["pred"] = np.argmax(metrics[["preds_1", "preds_2", "preds_3", "preds_4", "preds_5"]].values, axis=1) + 1
        grads = np.sqrt(np.sum(grads[0]**2, axis=0) / len(grads[0])) / np.sqrt(np.sum(grads[0]**2, axis=0) / len(grads[0])).max()
        return metrics["pred"], grads

    def run(self, *,
            archive_path=None,
            doc_markup_path=None,
            ai_markup_path=None,
            scan_path=None):
        logger.info('===== Compare markup: =====')
        logger.info('Input parameters:')
        if archive_path is not None:
            logger.info(f'archive_path: {archive_path}')
            logger.info(f'archive_path exist? {os.path.isfile(archive_path)}')
        else:
            logger.info(f'doc_markup_path: {doc_markup_path}')
            logger.info(f'doc_markup_path exist? {os.path.isfile(doc_markup_path)}')
            logger.info(f'doc_markup_path: {doc_markup_path}')
            logger.info(f'doc_markup_path exist? {os.path.isfile(doc_markup_path)}')
            logger.info(f'ai_markup_path: {ai_markup_path}')
            logger.info(f'ai_markup_path exist? {os.path.isfile(ai_markup_path)}')
            logger.info(f'scan_path: {scan_path}')
            logger.info(f'scan_path exist? {os.path.isfile(scan_path)}')
        logger.info('====================')

        if archive_path is not None:
            return {
                'rating': 42,
                'metrics': [.42] * 50
            }

        self.update_state(state='PROGRESS', meta={'ping': 'pong'})
        time.sleep(random.randint(2, 5))

        metrics = self.get_metrics_table(doc_markup_path, ai_markup_path)
        result, grads = self.predict(metrics, 'p.npy')

        labels = [
            "ACC Macro",
            "Bangdiwala B",
            "Bennett S",
            "Conditional Entropy",
            "Cross Entropy",
            "F1 Micro",
            "FNR Micro",
            "FPR Micro",
            "Gwet AC1",
            "Hamming Loss",
            "Joint Entropy",
            "Kappa No Prevalence",
            "Mutual Information",
            "NIR",
            "Overall ACC",
            "Overall RACC",
            "Overall RACCU",
            "PPV Micro",
            "Reference Entropy",
            "Response Entropy",
            "Standard Error",
            "TNR Micro",
            "TPR Micro"
        ]

        return {
            'rating': int(result.values[0]),
            'metrics': grads.tolist(),
            'labels': labels
        }


app = Celery(
    'tasks',
    backend=os.getenv('CELERY_BACKEND_URL', 'redis://localhost'),
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://localhost')
)
app.register_task(CompareMarkups())
