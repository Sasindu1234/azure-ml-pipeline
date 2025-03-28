from azure.identity import DefaultAzureCredential,ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace, ComputeInstance, AmlCompute
import datetime
from azure.storage.blob import BlobServiceClient
import json
from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml.entities import AccountKeyConfiguration
from azure.ai.ml.entities import Environment
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
import os
import time

def create_storage(TENANT_ID, CLIENT_ID, CLIENT_SECRET,subscription_id, resource_group,storage_account_name,location="eastus"):
    credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
    resource_client = ResourceManagementClient(credential, subscription_id)
    storage_client = StorageManagementClient(credential, subscription_id)
    
    resource_client.resource_groups.create_or_update(
        resource_group,
        {"location": location}
    )

    storage_account = storage_client.storage_accounts.begin_create(
        resource_group,
        storage_account_name,
        {
            "sku": {"name": "Standard_LRS"},
            "kind": "StorageV2",
            "location": location,
            "enable_https_traffic_only": True,
            "minimum_tls_version": "TLS1_2"
        }
    ).result()
    print(f"Storage account created: {storage_account.name}")
    keys = storage_client.storage_accounts.list_keys(
        resource_group, 
        storage_account_name
    )
    
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={storage_account_name};"
        f"AccountKey={keys.keys[0].value};"
        f"EndpointSuffix=core.windows.net"
    )

    return {
        "account_name": storage_account_name,
        "primary_key": keys.keys[0].value,
        "connection_string": connection_string
    }

    pass
    


def create_ml_resources(subscription_id, resource_group,workspace_name,compute_instance_name,compute_cluster_name,TENANT_ID, CLIENT_ID, CLIENT_SECRET, location="eastus"):
    # Initialize MLClient
    credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
    
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Create compute instance
    
    compute_instance = ComputeInstance(
        name=compute_instance_name,
        size="Standard_E4ds_v4",
    )
    ml_client.begin_create_or_update(compute_instance).result()
    print(f"Compute instance '{compute_instance_name}' created.")

    # Create AML compute cluster
   
    compute_cluster = AmlCompute(
        name=compute_cluster_name,
        type="amlcompute",
        size="Standard_D4s_v3",
        location=location,
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=60,
    )
    ml_client.begin_create_or_update(compute_cluster).result()
    print(f"Compute cluster '{compute_cluster_name}' created.")

    


def create_containers_and_upload_files(connect_str, container_names, tenant_ids, local_file_paths):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    for container_name in container_names:
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
            print(f"Container '{container_name}' created.")
        else:
            print(f"Container '{container_name}' already exists.")

    container_client = blob_service_client.get_container_client("rawdata")
    for tenant_id in tenant_ids:
        tenant_folder = f"{tenant_id}/"
        for file_name, local_path in local_file_paths.items():
            blob_client = container_client.get_blob_client(tenant_folder + file_name)
            if not blob_client.exists():  # Prevents error if blob already exists
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data)
                print(f"Uploaded '{file_name}' for tenant {tenant_id}.")
            else:
                print(f"Blob '{file_name}' already exists for tenant {tenant_id}, skipping upload.")



def create_datastore(subscription_id, resource_group, account_keyvalue,workspace_name):
    
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    store = AzureBlobDatastore(
    name="raw_data",
    description="Datastore created with rawdata container",
    account_name="leaveblob",
    container_name="rawdata",
    protocol="https",
    credentials=AccountKeyConfiguration(
        account_key= account_keyvalue
    ),)
    ml_client.create_or_update(store)

    store = AzureBlobDatastore(
    name="result_store",
    description="Datastore created with rawdata result",
    account_name="leaveblob",
    container_name="result",
    protocol="https",
    credentials=AccountKeyConfiguration(
        account_key= account_keyvalue
    ),
)
    
    ml_client.create_or_update(store)

    store = AzureBlobDatastore(
        name="pre_prodata",
        description="Datastore created with preprodata container",
        account_name="leaveblob",
        container_name="preprodata",
        protocol="https",
        credentials=AccountKeyConfiguration(
            account_key= account_keyvalue
        ),
    )

    ml_client.create_or_update(store)


def create_environment(subscription_id, resource_group,workspace_name):

    
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)


    # Define the environment
    env_name = "clustertestingenvironment"
    env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        name=env_name,
        conda_file="conda.yaml",
        description="Environment for clustering testing",
    )

    # Register the environment
    ml_client.environments.create_or_update(env)
    print(f"Environment '{env_name} registered.")

    

# Example usage
if __name__ == "__main__":
    # Load configuration from config.json
    config = {
        "TENANT_ID" : os.getenv("AZURE_TENANT_ID"),
        "CLIENT_ID" : os.getenv("AZURE_CLIENT_ID"),
        "CLIENT_SECRET" : os.getenv("AZURE_CLIENT_SECRET"),
        "subscription_id": os.getenv("SUBSCRIPTION_ID"),
        "resource_group": os.getenv("RESOURCE_GROUP"),
        "workspace_name": os.getenv("WORKSPACE_NAME"),
        "location": os.getenv("LOCATION"),
        "connect_str": os.getenv("CONNECT_STR"),
        "container_names": ["rawdata", "preprodata", "result"],
        "tenant_ids": ["tenant1", "tenant2"],  # Update with your tenant IDs
        "local_file_paths": {
            "query1.csv": "data/query1.csv",  # Update with your actual paths
            "query2.csv": "data/query2.csv"
        },
        "account_keyvalue": os.getenv("ACCOUNT_KEYVALUE"),
        "countainer_namerun": "rawdata",
        "compute_instance_name": "sasindu6",
        "compute_cluster_name" : "testone"
    }

    workspace_name = config["workspace_name"] 

        # Call functions with configuration values
    print(f"Creating ML resources for workspace '{workspace_name}'...")
    # Call functions with configuration values
    
    
    create_ml_resources(
        subscription_id=config["subscription_id"],
        resource_group=config["resource_group"],
        workspace_name= workspace_name,
        compute_instance_name = config["compute_instance_name"],
        compute_cluster_name = config["compute_cluster_name"],
        TENANT_ID = config["TENANT_ID"],
        CLIENT_ID = config["CLIENT_ID"],
        CLIENT_SECRET = config["CLIENT_SECRET"],
        location=config["location"],
    )
    

    create_containers_and_upload_files(
        connect_str=config["connect_str"],
        container_names=config["container_names"],
        tenant_ids=config["tenant_ids"],
        local_file_paths=config["local_file_paths"],
    )

    create_datastore(
        subscription_id=config["subscription_id"],
        resource_group=config["resource_group"],
        account_keyvalue=config["account_keyvalue"],
        workspace_name= workspace_name
    )

    create_environment(
        subscription_id=config["subscription_id"],
        resource_group=config["resource_group"],
        workspace_name= workspace_name
    )