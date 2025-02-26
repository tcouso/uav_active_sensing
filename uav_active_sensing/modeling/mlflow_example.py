import os
import mlflow
from pathlib import Path

# Define a folder to store artifacts
artifact_dir = Path("./mlflow_artifacts")  # or use any path you'd like
artifact_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist

# Create a dummy file to log as an artifact
dummy_file_path = artifact_dir / "dummy_model.txt"

# Create a dummy model (just a text file in this case)
with open(dummy_file_path, 'w') as f:
    f.write("This is a dummy model file.")

# Start an MLflow run
with mlflow.start_run():
    # Log the artifacts (the entire directory in this case)
    mlflow.log_artifacts(artifact_dir, artifact_path="models")  # This will store the file in MLflow

    # Log a metric for demonstration
    mlflow.log_metric("dummy_metric", 1)

    # Print out the run id
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# Check where the artifacts are stored
print(f"Artifacts are logged to: {artifact_dir}")
