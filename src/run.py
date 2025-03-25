import subprocess
import time
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def create_client():
    # Initialize MLClient
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )
    return ml_client

ml_client = create_client()

def all_jobs_completed():
    """Check if all Azure ML jobs are completed."""
    jobs = list(ml_client.jobs.list()) 
    for job in jobs:
        print(f"Job Name: {job.name}, Status: {job.status}")
        if job.status not in ["Completed", "Failed", "Canceled"]:
            return False  
    return True  

if __name__ == "__main__":
    print("Running script1.py...")
    subprocess.run(["python", "src/resourcecreate.py"], check=True)

    print("Waiting for all Azure ML jobs to complete...")
    while not all_jobs_completed():
        time.sleep(60) 

    print("All jobs completed! Running script2.py...")
    subprocess.run(["python", "src/runpipline.py"], check=True)