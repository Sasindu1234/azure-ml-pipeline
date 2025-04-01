import pandas as pd 
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import argparse
def GetFinalClusters(df, dict_leave_years):
   
    # Process df_leave_count_withoutyear (default year = 0)
    final_clusters_without_year = process_clusters(df, year=0)

    # Process each year in dict_leave_years separately
    final_clusters_by_year = {}
    for year, data in dict_leave_years.items():
        final_clusters_by_year[year] = process_clusters(data, year)

    return final_clusters_without_year, final_clusters_by_year

def process_clusters(data, year):
    
    # Step 1: Perform clustering scaling
    clusters = Clusterscaling(data)
    
    # Step 2: Perform deep clustering
    deep_clusters = DeepCluster(clusters, data)
    
    # Add year column to the result
    deep_clusters["Year"] = year
    
    # Return the relevant columns
    return deep_clusters[['EmployeeCode', 'Pattern','Level1', 'Level2', 'Level3', 'Year', 'SilhouteScore', 'SilhouteScore2', 'SilhouteScore3']]


def Clusterscaling(Data):
    np.random.seed(42)  # Ensures reproducibility

    # Remove EmployeeCode and convert to float
    X = np.array(Data.drop(['EmployeeCode'], axis=1).astype(float))

    range_n_clusters = [2, 3]
    silhouette_scores = {}

    # If dataset is too small, return "No Groups"
    if Data.shape[0] <= 20:
        return "No Groups"

    # Apply MinMax Scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)
    
    # Clustering with MinMax scaling
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        cluster_labels = kmeans.labels_
        
        score = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores[num_clusters] = score
        
        
    # Find the best number of clusters
    clusters = list(silhouette_scores.items())
    clusters.sort(key=lambda x: x[1], reverse=True)

    (best_n, best_score) = clusters[0]
    second_n, second_score = clusters[1] if len(clusters) > 1 else (None, None)


    # If the highest is > 0.85, check the second value
    if best_score > 0.85:
        if second_score is not None and second_score > 0.85:
            best_n = min(best_n, second_n)  # Select the cluster with the smaller number
            best_score = silhouette_scores[best_n]
        elif second_score is not None and 0.3 < second_score < 0.85:
            best_n = second_n
            best_score = second_score
    
    final_kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10)
    final_kmeans.fit(scaled_features)

    return final_kmeans.labels_, best_score

def ExtractClustersCounts(Clusters,Data):
    FinalData = {}
    for cluster in np.unique(Clusters[0]):
        index = np.where(Clusters[0] == cluster)
        tempData = Data.iloc[index]
        FinalData[cluster] = tempData
    return FinalData

import pandas as pd

def process_silhoute_score(data):
    score = Clusterscaling(data)[1]
    if isinstance(score, str):
        no_groups = {'o': 0.0, 'N': 0.0}
        return no_groups.get(score, 0.0)
    return score

def DeepCluster(Clusters, Data):
    if isinstance(Clusters[0], str):  # If clustering is not applicable
        print("No Clustering Needed")
        return pd.DataFrame({
            'EmployeeCode': Data['EmployeeCode'],
            'Level1': "1",
            'Level2': "1",
            'Level3': "1",
            'SilhouteScore': process_silhoute_score(Data)
        })

    # Level 1 Clustering
    Level1 = ExtractClustersCounts(Clusters, Data)
    treeTable = pd.DataFrame(columns=["EmployeeCode", "Level1", "SilhouteScore"])

    level1_dict = {}  # Mapping for Level1 Clusters
    for idx, (cluster, cluster_data) in enumerate(Level1.items(), start=1):
        level1_dict[cluster] = idx  # Assign unique Level1 ID
        
        tempData = pd.DataFrame({
            'EmployeeCode': cluster_data['EmployeeCode'],
            'Level1': str(idx),  
            'SilhouteScore': Clusters[1]
            
        })
        treeTable = pd.concat([treeTable, tempData], ignore_index=True)

    # Level 2 Clustering
    level2Clust = pd.DataFrame(columns=["EmployeeCode", "Level2"])
    intermediateSet = {}

    for cluster, cluster_data in Level1.items():
        Level2 = Clusterscaling(cluster_data)
        #print(f"Level 1 Cluster {cluster} - Size: {len(cluster_data)}")

        if isinstance(Level2, str):  # No further clustering
            tempData = pd.DataFrame({
                'EmployeeCode': cluster_data['EmployeeCode'],
                'Level2': "1",
                'SilhouteScore2': 0
                
            })
            intermediateSet[len(intermediateSet)] = cluster_data
        else:
            tempExtract = ExtractClustersCounts(Level2, cluster_data)
            tempDataList = []
            for sub_idx, (sub_cluster, sub_data) in enumerate(tempExtract.items(), start=1):
                tempDf = pd.DataFrame({
                    'EmployeeCode': sub_data['EmployeeCode'],
                    'Level2': str(sub_idx),
                    'SilhouteScore2': Level2[1]
                })
                tempDataList.append(tempDf)
                intermediateSet[len(intermediateSet)] = sub_data  

            tempData = pd.concat(tempDataList, ignore_index=True)
        level2Clust = pd.concat([level2Clust, tempData], ignore_index=True)

    treeTable = treeTable.merge(level2Clust, on='EmployeeCode', how='left')

    # Level 3 Clustering
    level3Clust = pd.DataFrame(columns=["EmployeeCode", "Level3"])
    finalSet = {}

    for cluster, cluster_data in intermediateSet.items():
        #print(f"Level 2 Cluster {cluster} - Size: {len(cluster_data)}")
        Level3 = Clusterscaling(cluster_data)

        if isinstance(Level3, str):
            tempData = pd.DataFrame({
                'EmployeeCode': cluster_data['EmployeeCode'],
                'Level3': "1",
                'SilhouteScore3': 0
            })
        else:
            tempExtract = ExtractClustersCounts(Level3, cluster_data)
            tempDataList = []
            for sub_idx, (sub_cluster, sub_data) in enumerate(tempExtract.items(), start=1):
                parent_level2 = treeTable.loc[treeTable['EmployeeCode'].isin(sub_data['EmployeeCode']), 'Level2'].values[0]
                tempDf = pd.DataFrame({
                    'EmployeeCode': sub_data['EmployeeCode'],
                    'Level3': str(sub_idx),
                    'SilhouteScore3': Level3[1]
                })
                tempDataList.append(tempDf)

            tempData = pd.concat(tempDataList, ignore_index=True)
        level3Clust = pd.concat([level3Clust, tempData], ignore_index=True)

    treeTable = treeTable.merge(level3Clust, on='EmployeeCode', how='left')
    # Add the pattern numbering system here
    treeTable['Order'] = (treeTable.Level1.astype(str) + 
                         treeTable.Level2.astype(str) + 
                         treeTable.Level3.astype(str)).astype(int)
    
    # Get unique patterns and assign sequential numbers
    Patterns = np.sort(treeTable['Order'].unique())
    order_mapping = pd.DataFrame({
        'Order': Patterns,
        'Pattern': np.arange(1, len(Patterns) + 1 ) # Start numbering from 1
    })
    
    # Merge the pattern numbers back to the main table
    treeTable = treeTable.merge(order_mapping, on='Order', how='left')
    
    
    
    

    return treeTable




def save_clusters_combined(clusters_without_year, clusters_by_year, file_prefix, tenant_id, base_folder="results"):
    """
    Save clustering results as two consolidated files:
    - One file without years
    - One file with all years combined
    """
    # Define the tenant-specific folder
    folder = os.path.join(base_folder, tenant_id)
    os.makedirs(folder, exist_ok=True)

    # Combine all years into a single DataFrame
    if isinstance(clusters_by_year, dict):
        combined_df = pd.concat(clusters_by_year.values(), ignore_index=True)
        combined_df = pd.concat([clusters_without_year, combined_df], ignore_index=True)
        combined_df.to_csv(os.path.join(folder, f"{file_prefix}.csv"), index=False)
    else:
        raise ValueError("Clusters by year must be a dictionary of DataFrames.")
        

def main(args):
    # Load preprocessed data for leave type clustering
    leave_type_data_path = os.path.join(args.leave_type_data_path,args.tenant_id,"leave_type_clustering_data.csv")
    leave_type_dict_path = os.path.join(args.leave_type_dict_path,args.tenant_id,  "leave_type_clustering_dict.pkl")
    print(f"Looking for leave type data at: {leave_type_data_path}")
    
    if not os.path.exists(leave_type_data_path):
        print(f"File not found: {leave_type_data_path}")
        return
    leave_type_data = pd.read_csv(leave_type_data_path)
    with open(leave_type_dict_path, "rb") as f:
        leave_type_dict = pickle.load(f)

    # Perform clustering for leave type
    leave_type_clusters_without_year, leave_type_clusters_by_year = GetFinalClusters(leave_type_data, leave_type_dict)
    
    # Save leave type clustering results in the tenant-specific folder
    save_clusters_combined(leave_type_clusters_without_year, leave_type_clusters_by_year, "leave_type_clusters", args.tenant_id,args.result_folder)

    # Load preprocessed data for date clustering
    date_clustering_data_path = os.path.join(args.date_clustering_data_path,args.tenant_id,  "date_clustering_data.csv")
    date_clustering_dict_path = os.path.join(args.date_clustering_dict_path,args.tenant_id, "date_clustering_dict.pkl")
    
    date_clustering_data = pd.read_csv(date_clustering_data_path)
    with open(date_clustering_dict_path, "rb") as f:
        date_clustering_dict = pickle.load(f)

    # Perform clustering for date clustering
    date_clusters_without_year, date_clusters_by_year = GetFinalClusters(date_clustering_data, date_clustering_dict)
    
    # Save date clustering results in the tenant-specific folder
    save_clusters_combined(date_clusters_without_year, date_clusters_by_year, "date_clusters", args.tenant_id, args.result_folder)


def parse_args():
    # Setup arg parser to get file paths
    parser = argparse.ArgumentParser()

    # Add arguments for file paths
    parser.add_argument("--leave_type_data_path", type=str, required=True, help="Path to the leave type data CSV")
    parser.add_argument("--leave_type_dict_path", type=str, required=True, help="Path to the leave type clustering dictionary")
    parser.add_argument("--date_clustering_data_path", type=str, required=True, help="Path to the date clustering data CSV")
    parser.add_argument("--date_clustering_dict_path", type=str, required=True, help="Path to the date clustering dictionary")
    parser.add_argument("--result_folder", dest='result_folder', type=str, required=True, help="Folder to save clustering results")
    parser.add_argument("--tenant_id", type=str, required=True, help="Tenant ID (folder name to save clustering results)")
    
    # Parse args
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    # run main function
    main(args)
   

