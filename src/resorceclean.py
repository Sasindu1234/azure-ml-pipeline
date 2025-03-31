from azure.mgmt.storage import StorageManagementClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,ClientSecretCredential
import os

def delete_data_store(TENANT_ID, CLIENT_ID, CLIENT_SECRET, subscription_id, resource_group_name,workspace_name,datastore_names):
    credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
    ml_client = MLClient(credential, subscription_id, resource_group_name, workspace_name)

    for datastore_name in datastore_names:
        try:
            ml_client.datastores.delete(datastore_name)
            print(f"Successfully deleted datastore: {datastore_name}")
        except Exception as e:
            print(f"Error deleting datastore {datastore_name}: {str(e)}")


def delete_storage_account(TENANT_ID, CLIENT_ID, CLIENT_SECRET, subscription_id, resource_group_name, storage_account_name):
    credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
    storage_client = StorageManagementClient(credential, subscription_id)
    
    print(f"Deleting storage account: {storage_account_name}...")
    storage_client.storage_accounts.delete(resource_group_name, storage_account_name)
    print("Storage account deleted successfully")




def delete_compute_cluster(TENANT_ID, CLIENT_ID, CLIENT_SECRET, subscription_id, resource_group_name, workspace_name, compute_name):
    credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
    ml_client = MLClient(credential, subscription_id, resource_group_name, workspace_name)
    
    print(f"Deleting compute cluster: {compute_name}...")
    ml_client.compute.begin_delete(compute_name).wait()
    print("Compute cluster deleted successfully")


if __name__ == "__main__":
    # Load configuration from config.json
    config = {
        "TENANT_ID" : os.getenv("AZURE_TENANT_ID"),
        "CLIENT_ID" : os.getenv("AZURE_CLIENT_ID"),
        "CLIENT_SECRET" : os.getenv("AZURE_CLIENT_SECRET"),
        "subscription_id": os.getenv("SUBSCRIPTION_ID"),
        "resource_group": os.getenv("RESOURCE_GROUP"),
        "workspace_name": os.getenv("WORKSPACE_NAME"),
        "storage_account_name" : os.getenv("STORAGE_NAME"),
        "compute_instance_name": "sasindu13",
        "datastore_names": ["raw_data", "pre_prodata", "result_store"]
       
    }

    delete_data_store(
        TENANT_ID = config["TENANT_ID"],
        CLIENT_ID = config["CLIENT_ID"],
        CLIENT_SECRET = config["CLIENT_SECRET"],
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name= config["workspace_name"],
        datastore_names = config["datastore_names"]
    )
    
    delete_storage_account(
        TENANT_ID = config["TENANT_ID"],
        CLIENT_ID = config["CLIENT_ID"],
        CLIENT_SECRET = config["CLIENT_SECRET"],
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        storage_account_name = config["storage_account_name"]
    )
  
    delete_compute_cluster(
        TENANT_ID = config["TENANT_ID"],
        CLIENT_ID = config["CLIENT_ID"],
        CLIENT_SECRET = config["CLIENT_SECRET"],
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name= config["workspace_name"], 
        compute_name = config["compute_instance_name"]
        
    )