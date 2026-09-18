# start from python base image
FROM python:3.14

# change working directory


# add requirements file to image
COPY ./requirements.txt .

# install python libraries
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# add python code
COPY ./fast_api.py .
COPY ./models ./models

# specify default commands
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "5999"]