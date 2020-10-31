#!bin/bash

export C_FORCE_ROOT="true"

cd /celery_tasks
celery -A tasks worker --loglevel=info
