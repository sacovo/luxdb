from python:3.9

WORKDIR /usr/src/luxdb

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY luxdb ./luxdb/

COPY luxdb-server LICENSE ./

RUN groupadd -g 1020 -r luxdb && \
    useradd -u 1020 --no-log-init -r -g luxdb luxdb && \
    mkdir /data/ && \
    chown -R luxdb:luxdb /data/

USER luxdb

VOLUME ["/data/"]

EXPOSE 8484

ENTRYPOINT ["./luxdb-server", "/data/db.pickle"]

CMD ["--loglevel", "WARNING", "--host", "0.0.0.0", "--port", "8484"]
