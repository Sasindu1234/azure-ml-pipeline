from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AccountKeyConfiguration
from azure.ai.ml import load_component as load_component_from_yaml
import json
import os

def create_client(subscription_id, resource_group, workspace_name):
    # Initialize MLClient
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    return ml_client


def load_components():
    prep_data = load_component_from_yaml(source="prep-data.yml")
    cluster_training = load_component_from_yaml(source="train-model.yml")
    return prep_data, cluster_training


def create_tenant_folders(connect_str, container_name):
    # Initialize BlobServiceClient and ContainerClient
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    tenant_folders = set()
    tenant_data_paths = {}

    # Loop through blobs in the container and extract tenant folders
    for file in container_client.walk_blobs(delimiter="/"):
        tenant_folder = file.name.split("/")[0]
        tenant_folders.add(tenant_folder)

    raw_data_base_path = "azureml://datastores/raw_data/paths"
    result_base_path = "azureml://datastores/result_store/paths"

    # Generate paths for each tenant folder
    for tenant_folder in tenant_folders:
        tenant_data_paths[tenant_folder] = {
            "input_data1": f"{raw_data_base_path}/{tenant_folder}/query1.csv",
            "input_data2": f"{raw_data_base_path}/{tenant_folder}/query2.csv",
        }
    return tenant_data_paths




def main():
    # Define variables
    
   
    config = {
        "subscription_id": os.getenv("SUBSCRIPTION_ID"),
        "resource_group": os.getenv("RESOURCE_GROUP"),
        "workspace_name": os.getenv("WORKSPACE_NAME"),
        "connect_str": os.getenv("CONNECT_STR"),
        "countainer_namerun": "rawdata",
        "compute_instance_name": "newcompute",
        "compute_cluster_name" : "testone"
    }

    # Create client
    ml_client = create_client(subscription_id=config["subscription_id"],
        resource_group=config["resource_group"],
        workspace_name= config["workspace_name"])

    # Load components
    prep_data, cluster_training = load_components()
    
    @pipeline()
    def clustering_pipeline(input_data1, input_data2, tenant_id):
        # Preprocessing the data
        clean_data = prep_data(
            input_data1=input_data1,
            input_data2=input_data2,
            tenant_id=tenant_id
        )


        clean_data.outputs.leave_type_clustering_data = Output(
            type="uri_folder",
            path=f"azureml://datastores/pre_prodata/paths/",
            mode="rw_mount"
        )
        clean_data.outputs.leave_type_clustering_dict = Output(
            type="uri_folder",
            path=f"azureml://datastores/pre_prodata/paths/",
            mode="rw_mount"
        )
        clean_data.outputs.date_clustering_data = Output(
            type="uri_folder",
            path=f"azureml://datastores/pre_prodata/paths/",
            mode="rw_mount"
        )
        clean_data.outputs.date_clustering_dict = Output(
            type="uri_folder",
            path=f"azureml://datastores/pre_prodata/paths/",
            mode="rw_mount"
        )

        # Training the clustering model
        train_model = cluster_training(
            leave_type_data_path=clean_data.outputs.leave_type_clustering_data,
            leave_type_dict_path=clean_data.outputs.leave_type_clustering_dict,
            date_clustering_data_path=clean_data.outputs.date_clustering_data,
            date_clustering_dict_path=clean_data.outputs.date_clustering_dict,
            tenant_id=tenant_id
        )

        # Store results in a dynamic folder based on tenant_id
        train_model.outputs.clustered_results = Output(
            type="uri_folder",
            path=f"azureml://datastores/result_store/paths/",
            mode="rw_mount"
        )

        return {
            "clustered_results": train_model.outputs.clustered_results,
        }

    # Create tenant folders
    tenant_data_paths = create_tenant_folders(
        connect_str = config["connect_str"], 
        container_name = config["countainer_namerun"])

    # Loop through each tenant and submit pipeline jobs
    for tenant_id, paths in tenant_data_paths.items():
        # Create pipeline job for the tenant
        pipeline_job = clustering_pipeline(
            input_data1=Input(type=AssetTypes.URI_FILE, path=paths["input_data1"]),
            input_data2=Input(type=AssetTypes.URI_FILE, path=paths["input_data2"]),
            tenant_id=tenant_id,
        )

        # Set pipeline-level compute and datastore
        pipeline_job.settings.default_compute = config["compute_instance_name"]
        pipeline_job.settings.default_datastore = "workspaceblobstore"

        # Submit the pipeline job to the workspace
        submitted_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name=f"pipeline_cluster_{tenant_id}"
        )
        print(f"Submitted job for tenant {tenant_id}: {submitted_job.name}")


if __name__ == "__main__":
    main()