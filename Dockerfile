FROM tensorflow/tensorflow

WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless

COPY model.py .
COPY model_08015_07680.h5 .
COPY images /usr/src/app/images
COPY labels.txt .
COPY test_run.py .



CMD ["python","./test_run.py"]