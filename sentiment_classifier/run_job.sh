# Variables
cluster_name=training
registry_name=k3d-registry
registry_port=5050  # Change this to the default Docker registry port
image_name=$registry_name:$registry_port/my-training-job:latest

k3d cluster create training --agents 2 --registry-create training-registry

# Copy pyproject.toml and poetry.lock
cp ../pyproject.toml .
cp ../poetry.lock .

# Build the Docker image
docker build -t $image_name .

# Push the Docker image to the registry
docker push $image_name

# Run the Kubernetes job
kubectl delete job my-training-job
sed "s|\${IMAGE_NAME}|$image_name|g" ./job.yaml | kubectl apply -f -

# Print out the kubeconfig file
echo "Kubeconfig file for the $cluster_name cluster:"
k3d kubeconfig get $cluster_name

# Delete pyproject.toml and poetry.lock
rm pyproject.toml
rm poetry.lock