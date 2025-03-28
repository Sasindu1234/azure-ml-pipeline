import subprocess
import time
import os
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

def create_client():
    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET")
    )
    return MLClient(
        credential,
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )

def check_environment_ready(ml_client, env_name="clustertestingenvironment", timeout=600):
    """Check if environment is successfully created"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            env = ml_client.environments.get(env_name, label="latest")
            if env:
                print(f"Environment '{env_name}' is ready")
                return True
        except Exception as e:
            print(f"Environment not ready yet: {str(e)}")
        time.sleep(30)
    return False

if __name__ == "__main__":
    # Step 1: Run resource creation and verify completion
    print("Creating resources...")
    subprocess.run(["python", "src/resourcecreate.py"], check=True)
    
    # Step 2: Verify environment is ready
    print("Verifying environment...")
    ml_client = create_client()
    if not check_environment_ready(ml_client):
        raise RuntimeError("Environment creation failed or timed out")
    
    # Step 3: Run pipeline only after verification
    print("Running pipeline...")
    pipeline_process = subprocess.run(
        ["python", "src/runpipline.py"],
        check=True,
        capture_output=True,
        text=True
    )
    print("Pipeline execution completed")