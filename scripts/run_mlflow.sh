echo "MLflow server will be accessible at: http://0.0.0.0:5001"

mkdir -p ~/mlruns

mlflow server \
    --backend-store-uri ~/mlruns \
    --default-artifact-root ~/mlruns \
    --host 0.0.0.0 \
    --port 5001