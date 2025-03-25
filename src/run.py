import subprocess
import time
import os
import sys
from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def create_client():
    try:
        credential = DefaultAzureCredential(
            additionally_allowed_tenants=["*"],
            exclude_interactive_browser_credential=False
        )
        
        ml_client = MLClient(
            credential,
            subscription_id=os.getenv("SUBSCRIPTION_ID"),
            resource_group=os.getenv("RESOURCE_GROUP"),
            workspace_name=os.getenv("WORKSPACE_NAME")
        )
        return ml_client
    except Exception as e:
        print(f"Failed to create ML client: {str(e)}")
        sys.exit(1)

def all_jobs_completed(ml_client, timeout_minutes=60):
    """Check if all Azure ML jobs are completed with timeout."""
    start_time = time.time()
    while True:
        jobs = list(ml_client.jobs.list())
        all_done = True
        
        for job in jobs:
            print(f"Job Name: {job.name}, Status: {job.status}")
            if job.status not in ["Completed", "Failed", "Canceled"]:
                all_done = False
                break
        
        if all_done:
            return True
            
        if (time.time() - start_time) > (timeout_minutes * 60):
            print(f"Timeout reached after {timeout_minutes} minutes")
            return False
            
        time.sleep(60)

def run_script(script_name):
    """Run a Python script with proper path resolution."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        return False
        
    try:
        subprocess.run(["python", str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {str(e)}")
        return False

if __name__ == "__main__":
    ml_client = create_client()
    
    # Run resource creation
    if not run_script("resourcecreate.py"):
        sys.exit(1)
        
    # Monitor jobs
    if not all_jobs_completed(ml_client):
        print("Error: Jobs did not complete in expected time")
        sys.exit(1)
        
    # Run pipeline
    if not run_script("runpipline.py"):
        sys.exit(1)
        
    print("Pipeline completed successfully")