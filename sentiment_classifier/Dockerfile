FROM python:3.12-slim

WORKDIR /sentiment_classifier
RUN pip install --no-cache-dir poetry
COPY ./pyproject.toml ./poetry.lock ./
RUN poetry install --no-interaction --no-ansi

ADD . /sentiment_classifier

CMD ["poetry", "run", "python", "train.py"]