#!bin/bash

cd /app

if [ $USE_SOCK_FILE = "True" ]
then
    uvicorn main:app --uds hack2020-backend.sock
else
    uvicorn main:app --host 0.0.0.0 --port 80 --reload
fi
