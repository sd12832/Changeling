cp ../pyproject.toml .
cp ../poetry.lock .
docker build -t my-training-job .
rm pyproject.toml
rm poetry.lock
