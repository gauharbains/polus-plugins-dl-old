FROM labshare/polus-bfio-util:1.0.8-slim-buster

COPY VERSION /

ARG EXEC_DIR="/opt/executables"

#Create folders
RUN mkdir -p ${EXEC_DIR}

#Copy executable
COPY src ${EXEC_DIR}/

RUN pip install --upgrade setuptools

RUN pip install -r ${EXEC_DIR}/requirements.txt --no-cache-dir && \
    mv ${EXEC_DIR}/tensorflow_backend.py /usr/local/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py -f 

ENTRYPOINT [ "python3", "models.py" ]
