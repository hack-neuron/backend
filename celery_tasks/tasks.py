# coding: utf-8

import os
import random
import time

from celery import Celery, Task
from celery.utils.log import get_task_logger

path = os.path.dirname(__file__)
upload_path = os.path.join(path, '..', 'uploads')

logger = get_task_logger(__name__)


class CompareMarkups(Task):
    name = 'compare_markups'
    # throws = (ValueError,)

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

        return {
            'rating': random.random(),
            'metrics': [random.random() for _ in range(50)]
        }


app = Celery(
    'tasks',
    backend=os.getenv('CELERY_BACKEND_URL', 'redis://localhost'),
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://localhost')
)
app.register_task(CompareMarkups())
