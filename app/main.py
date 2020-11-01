# coding: utf-8

import os

import aiofiles
import uvicorn
from celery import Celery
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

path = os.path.dirname(__file__)
upload_path = os.path.join(path, '..', 'uploads')
static_path = os.path.join(path, 'static')

app = FastAPI(
    title='Neuramark API',
    version='1.0.0',
    docs_url=None,
    redoc_url=None
)

app.mount('/static', StaticFiles(directory=static_path), name='static')

celery_app = Celery(
    'tasks',
    backend=os.getenv('CELERY_BACKEND_URL', 'redis://localhost'),
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://localhost')
)


def get_task_by_uuid(uuid: str):
    """Достаёт задачу из очереди по заданному uuid."""
    return celery_app.AsyncResult(uuid)


@app.post('/upload')
async def upload(doc_markup: UploadFile = File(...),
                 ai_markup: UploadFile = File(...),
                 scan: UploadFile = File(...)):
    """Загрузка файлов на сервер.
    `doc_markup` -- разметка эксперта
    `ai_markup` -- разметка ИИ-сервиса
    `scan` -- рентгенограмма

    Отдаёт `id: str` задачи, поставленной в очередь на выполнение.
    Статус задачи можно узнать с помощью метода `get_status`.
    """
    for upload_file in (doc_markup, ai_markup, scan):
        filename = upload_file.filename
        contents = await upload_file.read()
        path = os.path.join(upload_path, filename)
        async with aiofiles.open(path, mode='wb') as f:
            await f.write(contents)

    kwargs = {
        'doc_markup_path': os.path.join(upload_path, doc_markup.filename),
        'ai_markup_path': os.path.join(upload_path, ai_markup.filename),
        'scan_path': os.path.join(upload_path, scan.filename)
    }
    f = celery_app.send_task('compare_markups', kwargs=kwargs)
    return {'id': f.id}


@app.post('/upload_many')
async def upload(archive_file: UploadFile = File(...)):
    """Загрузка архива с файлами на сервер.
    `archive_file` -- архив с разметкой

    Отдаёт `id: str` задачи, поставленной в очередь на выполнение.
    Статус задачи можно узнать с помощью метода `get_status`.
    """
    filename = archive_file.filename
    contents = await archive_file.read()
    path = os.path.join(upload_path, filename)
    async with aiofiles.open(path, mode='wb') as f:
        await f.write(contents)

    kwargs = {
        'archive_path': os.path.join(upload_path, archive_file.filename)
    }
    f = celery_app.send_task('compare_markups', kwargs=kwargs)
    return {'id': f.id}


@app.get('/get_status')
async def get_status(id_: str):
    """Получение статуса выполнения задачи оп её `id`."""
    task = get_task_by_uuid(id_)
    state = task.state
    response = {'state': state}
    if state == 'PROGRESS':
        response.update(task.info)
    elif state == 'SUCCESS':
        result = task.get(1)
        task.forget()
        response.update({'result': result})
    return response


@app.get('/docs', include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f'{app.title} - Swagger UI',
        swagger_js_url='/static/js/swagger-ui-bundle.js',
        swagger_css_url='/static/css/swagger-ui.css',
        swagger_favicon_url='/static/img/favicon.png'
    )


if __name__ == '__main__':
    uvicorn.run('main:app',
                host='127.0.0.1',
                port=8081,
                log_level='info',
                reload=True)
