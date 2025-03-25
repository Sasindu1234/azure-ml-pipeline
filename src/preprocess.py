import pandas as pd
import pickle
from azure.ai.ml import MLClient
import mltable
from azure.identity import DefaultAzureCredential
import argparse
import argparse


def load_data_asset(asset_name: str, version: str = "1"):
    """Load an Azure ML data asset as a Pandas DataFrame."""
    
    # Initialize MLClient
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    
    # Get the data asset
    data_asset = ml_client.data.get(asset_name, version=version)
    
    # Load as MLTable
    tbl = mltable.load(f'azureml:/{data_asset.id}')
    
    # Convert to Pandas DataFrame
    df = tbl.to_pandas_dataframe()
    
    return df

def load(path):
    df = pd.read_csv(path)

    return df
    
def preprocess_leave_data(df_with_year, df_without_year):
    # Drop rows where LeaveTypeName is missing
    df_with_year = df_with_year.dropna(subset=["LeaveTypeName"])
    df_without_year = df_without_year.dropna(subset=["LeaveTypeName"])

    # Convert StartDate and EndDate to datetime format
    for df in [df_with_year, df_without_year]:
        df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
        df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce")

        # Drop rows where EndDate is before StartDate
        df.dropna(subset=["StartDate", "EndDate"], inplace=True)
        df = df[df["EndDate"] >= df["StartDate"]]

        # Fill missing NoOfDays based on date difference
        df["NoOfDays"] = df["NoOfDays"].fillna((df["EndDate"] - df["StartDate"]).dt.days + 1)

    # Aggregate leave data for df_without_year
    df_leave_count_withoutyear = df_without_year.pivot_table(index="EmployeeCode",columns="LeaveTypeName",values="NoOfDays",aggfunc="sum",fill_value=0).reset_index()


    # Process LeaveYear-based aggregation
    dict_leave_years = {}
    for year, group in df_with_year.groupby("LeaveYear"):
        leave_count_yearly = group.pivot_table(index="EmployeeCode",columns="LeaveTypeName",values="NoOfDays",aggfunc="sum",fill_value=0).reset_index()
        dict_leave_years[year] = leave_count_yearly

    return df_leave_count_withoutyear, dict_leave_years

def calculate_durations(row):
        start_date, end_date = row["StartDate"], row["EndDate"]
        
        # Calculate the total duration (inclusive of weekends)
        total_duration = (end_date - start_date).days + 1
        
        # Calculate duration excluding weekends
        date_range = pd.date_range(start=start_date, end=end_date)
        weekdays = date_range[~date_range.weekday.isin([5, 6])]  # Exclude Saturday (5) and Sunday (6)
        weekdays_duration = len(weekdays)
        
        return pd.Series([total_duration, weekdays_duration])

def stageone_data_prepro(df_with_year, df_without_year):

    for df in [df_with_year, df_without_year]:
        df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
        df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce")

        
    
    # Step 1: Keep only valid dates (EndDate > StartDate)
    df_with_year = df_with_year[df_with_year["EndDate"] >= df_with_year["StartDate"]].dropna()
    df_without_year = df_without_year[df_without_year["EndDate"] >= df_without_year["StartDate"]].dropna()
   

    # Apply the duration calculation function
    df_without_year[['Duration', 'Duration_without_weekend']] = df_without_year.apply(calculate_durations, axis=1)
    df_with_year[['Duration', 'Duration_without_weekend']] = df_with_year.apply(calculate_durations, axis=1)

    # Filter rows where NoOfDays equals the duration or is 0.5 (half-day leave)
    
    df_without_year_filter = df_without_year[(df_without_year['NoOfDays'] == df_without_year['Duration']) | ((df_without_year['NoOfDays'] == 0.5) & (df_without_year['Duration'] == 1)) ]
    df_other_data = df_without_year[~((df_without_year['NoOfDays'] == df_without_year['Duration']) | ((df_without_year['NoOfDays'] == 0.5) & (df_without_year['Duration'] == 1)))]
    dict_leave_years = {}
    dict_other_data_years = {}
    for year, group in df_with_year.groupby("LeaveYear"):
       
        
        df_with_year_filter = group[(group['NoOfDays'] == group['Duration']) | ((group['NoOfDays'] == 0.5) & (group['Duration'] == 1))]
        dict_leave_years[year] = df_with_year_filter
        
        # Other leave data (not matching the condition)
        df_other_data_year = group[~((group['NoOfDays'] == group['Duration']) | ((group['NoOfDays'] == 0.5) & (group['Duration'] == 1)))]
        dict_other_data_years[year] = df_other_data_year
        
    return df_without_year_filter, dict_leave_years,df_other_data ,dict_other_data_years 


def calculate_leave_days(df):
    # Initialize weekday columns (Monday to Friday)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in weekdays:
        df[day] = 0

    # Initialize Saturday and Sunday columns (empty or zero)
    df["Saturday"] = 0
    df["Sunday"] = 0

    # Iterate over rows to count the number of each weekday in the date range
    for idx, row in df.iterrows():
        if pd.notna(row["StartDate"]) and pd.notna(row["EndDate"]):
            date_range = pd.date_range(start=row["StartDate"], end=row["EndDate"])
            # Filter out weekends (Saturday (5) and Sunday (6))
            weekday_counts = date_range[~date_range.dayofweek.isin([5, 6])].dayofweek.value_counts().to_dict()

            # Assign weekday counts to the corresponding columns
            for i, day in enumerate(weekdays):
                df.at[idx, day] = weekday_counts.get(i, 0)

    # Calculate total leave days without weekends
    df["TotalLeaveDays"] = df[weekdays].sum(axis=1)

    return df

import pandas as pd

def process_leave_data_filterdate(filterdata):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    all_days = weekdays + ["Saturday", "Sunday"]

    # Ensure StartDate and EndDate are datetime objects
    filterdata = filterdata.copy()
    

    # Initialize all weekday columns to 0
    for day in all_days:
        filterdata[day] = 0

    def assign_weekdays(row):
        leave_dates = pd.date_range(row["StartDate"], row["EndDate"])  # Generate all leave dates
        if len(leave_dates) == 0:
            return row  # Avoid division by zero
        
        leave_per_day = row["NoOfDays"] / len(leave_dates)  # Distribute leave equally

        for date in leave_dates:
            day_name = date.strftime("%A")  # Get weekday name
            if day_name in all_days:
                row[day_name] += leave_per_day  # Assign fraction or full leave
        return row

    # Apply leave assignment function
    filterdata = filterdata.apply(assign_weekdays, axis=1)

    # Compute total leave days
    filterdata["TotalLeaveDays"] = filterdata[all_days].sum(axis=1)

    return filterdata


def stagetwo_data_prepro(df_filter,dict_filter,df_other,dict_other):
    
    prepro_Stagetwo_dict= {}

# Iterate through the keys (years) and apply the functions
    for year in dict_filter.keys():
        if year in dict_other:  # Ensure the key exists in both dictionaries
        
                prepro_filter =process_leave_data_filterdate(dict_filter[year])
                prepro_unpattern = calculate_leave_days(dict_other[year])  
                prepro_stage1 =  pd.concat([prepro_filter,  prepro_unpattern], axis=0)
                prepro_stage1 = prepro_stage1.drop(columns=['LeaveYear']) 
                prepro_Stagetwo_dict[year] = prepro_stage1
        elif year in dict_filter:
            # Process only dict_filter for the year
            prepro_filter = process_leave_data_filterdate(dict_filter[year])
            prepro_filter = prepro_filter.drop(columns=['LeaveYear']) 
            prepro_Stagetwo_dict[year] = prepro_filter
        elif year in dict_other:
            # Process only dict_other for the year
            prepro_unpattern = calculate_leave_days(dict_other[year])
            prepro_unpattern = prepro_unpattern.drop(columns=['LeaveYear'])
            prepro_Stagetwo_dict[year] = prepro_unpattern
        

    
    prepro_filter_df = process_leave_data_filterdate(df_filter)
    prepro_unpattern_df = calculate_leave_days(df_other)

    # Combine the processed DataFrames into one
    prepro_Stagetwo_df = pd.concat([prepro_filter_df, prepro_unpattern_df], axis=0)

    # Reset the index (optional)
    prepro_Stagetwo_df = prepro_Stagetwo_df.reset_index(drop=True)

    return prepro_Stagetwo_df,prepro_Stagetwo_dict

import pandas as pd

def preprocess_leave_day_sum(stagetwo_data):
   
   

    grouped_data = stagetwo_data.groupby('EmployeeCode')[
        ["NoOfDays", "Duration", "Duration_without_weekend", 
         "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
         "Saturday", "Sunday", "TotalLeaveDays"]
    ].sum().reset_index()
    prepro_Stagethree_df = grouped_data[['EmployeeCode', 'Monday', 'Tuesday', 'Wednesday', 
                                  'Thursday', 'Friday', 'Saturday', 'Sunday']]

    return prepro_Stagethree_df


def stagethree_data_prepro(stagetwo_df,stagetwo_dict):
    
    prepro_Stagethree_dict= {}

# Iterate through the keys (years) and apply the functions
    for year in stagetwo_dict.keys():
    
        
       prepro_stagethree =preprocess_leave_day_sum(stagetwo_dict[year])
       prepro_Stagethree_dict[year] = prepro_stagethree

    prepro_Stagethree_df = preprocess_leave_day_sum(stagetwo_df)
    
    # Reset the index (optional)
    prepro_Stagethree_df = prepro_Stagethree_df.reset_index(drop=True)

    return prepro_Stagethree_df,prepro_Stagethree_dict

def PreProcess_Main(query1, query2):
   
    # Stage 1 preprocessing
    df_filter, dict_filter, df_other, dict_other = stageone_data_prepro(query1, query2)
    
    # Stage 2 preprocessing
    stagetwo_df, stagetwo_dict = stagetwo_data_prepro(df_filter, dict_filter, df_other, dict_other)
    
    # Stage 3 preprocessing
    df, dict_leave_years = stagethree_data_prepro(stagetwo_df, stagetwo_dict)
    
    return df, dict_leave_years



def save_data(data, file_name, tenant_id, base_folder="processed"):
    """
    Save data to a file in the specified folder.
    """
    import os
    # Define the tenant-specific folder
    folder = os.path.join(base_folder, tenant_id)
    os.makedirs(folder, exist_ok=True)
    
    # Save DataFrame as CSV
    if isinstance(data, pd.DataFrame):
        data.to_csv(f"{folder}/{file_name}.csv", index=False)
    # Save dictionary as a pickle file
    elif isinstance(data, dict):
        with open(f"{folder}/{file_name}.pkl", "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError("Unsupported data type. Only DataFrame and dictionary are supported.")

def main(args):
    
    query1= load(args.input_data1)
    query2= load(args.input_data2)
    # Preprocess data for leave type clustering
    leave_type_data, leave_type_dict = preprocess_leave_data(query1, query2)
    
    # Preprocess data for date clustering
    date_clustering_data, dict_leave_years = PreProcess_Main(query1, query2)
    
    # Save all outputs
    
    save_data(leave_type_dict, "leave_type_clustering_dict", args.tenant_id, args.leave_type_clustering_dict)
    save_data(leave_type_data, "leave_type_clustering_data", args.tenant_id, args.leave_type_clustering_data)
    save_data(date_clustering_data, "date_clustering_data", args.tenant_id, args.date_clustering_data)
    save_data(dict_leave_years, "date_clustering_dict", args.tenant_id , args.date_clustering_dict)
    

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data1", dest='input_data1', type=str, required=True)
    parser.add_argument("--input_data2", dest='input_data2', type=str, required=True)
    parser.add_argument("--tenant_id", type=str, required=True)
    parser.add_argument("--leave_type_clustering_data", type=str, required=True)
    parser.add_argument("--leave_type_clustering_dict", type=str, required=True)
    parser.add_argument("--date_clustering_data", type=str, required=True)
    parser.add_argument("--date_clustering_dict", type=str, required=True)
    # parse args
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    

     # parse args
    args = parse_args()

    # run main function
    main(args)

