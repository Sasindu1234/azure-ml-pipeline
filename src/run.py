import subprocess
import time
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,ClientSecretCredential
from datetime import datetime

def create_client():
    # Explicitly get environment variables
    client_id = os.getenv("AZURE_CLIENT_ID")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    
    if not all([client_id, tenant_id, client_secret]):
        raise ValueError("Missing required Azure credentials in environment variables")
    
    credential = ClientSecretCredential( tenant_id, client_id,client_secret)
    
    ml_client = MLClient(
        credential,
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )
    return ml_client

  

if __name__ == "__main__":

    ml_client = create_client()
    print("Running script1.py...")
    subprocess.run(["python", "src/resourcecreate.py"], check=True)

    

    

    

    print("All jobs completed! Running script2.py...")
    subprocess.run(["python", "src/runpipline.py"], check=True)