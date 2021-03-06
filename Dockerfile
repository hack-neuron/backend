FROM python:3.8-slim

ADD script/entrypoint.sh /entrypoint.sh

ADD requirements.txt /requirements.txt

RUN \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
