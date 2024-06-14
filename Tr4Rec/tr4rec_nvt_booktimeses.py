# -*- coding: utf-8 -*-
"""


@author: 7000030999
"""
import sys
sys.path.append("/home/p/b/check/trans_2/lib/python3.10/site-packages")
import os
import glob
import numpy as np
import gc
import pandas as pd
import cudf
import cupy
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags

DATA_FOLDER=""
#FILENAME_PATTERN ='1.6_book_ts30dayssess_tr4recfinal1.csv'
FILENAME_PATTERN ='1.6_book_ts30dayssess_tr4rec1.csv'

DATA_PATH = os.path.join(DATA_FOLDER, FILENAME_PATTERN)

OVERWRITE = False

pandas_df=pd.read_csv(DATA_PATH)
"""
pandas_df = pd.read_csv(DATA_PATH, sep=',',
                                names=['item_id','timestamp', 'session_id'],
                                dtype={'item_id':'int','timestamp':'int64','session_id':'int'}, header=0)


interactions_df = cudf.read_csv(DATA_PATH, sep=',', 
                                names=['session_id','timestamp', 'item_id'], 
                                dtype=['int','int64', 'int'],header=0)
"""
interactions_df=cudf.from_pandas(pandas_df)
print("Count with in-session repeated interactions: {}".format(len(interactions_df)))
# Sorts the dataframe by session and timestamp, to remove consecutive repetitions
#interactions_df.timestamp = interactions_df.timestamp.astype(int)
interactions_df = interactions_df.sort_values(['session_id', 'timestamp'])
past_ids = interactions_df['item_id'].shift(1).fillna()
session_past_ids = interactions_df['session_id'].shift(1).fillna()
# Keeping only no consecutive repeated in session interactions
interactions_df = interactions_df[~((interactions_df['session_id'] == session_past_ids) & (interactions_df['item_id'] == past_ids))]
print("Count after removed in-session repeated interactions: {}".format(len(interactions_df)))

items_first_ts_df = interactions_df.groupby('item_id').agg({'timestamp': 'min'}).reset_index().rename(columns={'timestamp': 'itemid_ts_first'})
interactions_merged_df = interactions_df.merge(items_first_ts_df, on=['item_id'], how='left')
#interactions_merged_df.head(1000)
print(interactions_merged_df.head())

#interactions_merged_df.to_parquet(os.path.join(DATA_FOLDER, 'interactions_merged_mlmin_df.parquet'))
# print the total number of unique items in the dataset
print(interactions_merged_df['item_id'].nunique())

cat_feats = ColumnSelector(['item_id']) >> nvt.ops.Categorify(start_index=1)

# create time features
session_ts = ColumnSelector(['timestamp'])
session_time = (
    session_ts >> 
    nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
    nvt.ops.Rename(name = 'event_time_dt')
)
sessiontime_weekday = (
    session_time >> 
    nvt.ops.LambdaOp(lambda col: col.dt.weekday) >> 
    nvt.ops.Rename(name ='et_dayofweek')
)

# Derive cyclical features: Define a custom lambda function 
def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')

class ItemRecency(nvt.ops.Operator):
    def transform(self, columns, gdf):
        print(gdf.columns)
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['itemid_ts_first']
            delta_seconds = (col - item_first_timestamp).astype('int64')
            delta_days = delta_seconds / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf
    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        self._validate_matching_cols(input_schema, parents_selector, "computing input selector")
        return parents_selector

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[col_name + "_age_days"] = [col_name]
        return column_mapping

    @property
    def dependencies(self):
        return ["itemid_ts_first"]

    @property
    def output_dtype(self):
        return np.float64

recency_features = session_ts >> ItemRecency() 
# Apply standardization to this continuous feature
recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(name='product_recency_days_log_norm')

time_features = (
    session_time +
    sessiontime_weekday +
    weekday_sin + 
    recency_features_norm
)

features = ColumnSelector(['session_id', 'timestamp']) + cat_feats + time_features 
# Define Groupby Operator
groupby_features = features >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    sort_cols=["timestamp"],
    aggs={
        'item_id': ["list", "count"],
        #'category': ["list"],  
        'ts': ["first"],
        'event_time_dt': ["first"],
        'et_dayofweek_sin': ["list"],
        'product_recency_days_log_norm': ["list"]
        },
    name_sep="-")

# Truncate sequence features to first interacted 20 items 
SESSIONS_MAX_LENGTH = 20


item_feat = groupby_features['item_id-list'] >> nvt.ops.TagAsItemID()
cont_feats = groupby_features['et_dayofweek_sin-list', 'product_recency_days_log_norm-list'] >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])


groupby_features_list =  item_feat + cont_feats #+ groupby_features['category-list']
groupby_features_truncated = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=True)
# Calculate session day index based on 'event_time_dt-first' column

year_index = (
    groupby_features['event_time_dt-first'] >>
    nvt.ops.LambdaOp(lambda col: ((col - col.min()) / np.timedelta64(1, 'Y')).astype(int) + 1908 ) >>
    nvt.ops.Rename(f=lambda col: "year_index") >>
    nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
)

"""
day_index = ((groupby_features['event_time_dt-first'])  >> 
             nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
             nvt.ops.Rename(f = lambda col: "day_index") >>
             nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
            )

year_index = (
    groupby_features['event_time_dt-first'] >> 
    nvt.ops.LambdaOp(lambda col: ((col - col.min()) / np.timedelta64(1, 'Y')).astype(int) + 2008) >>
    nvt.ops.Rename(f=lambda col: "year_index") >>
    nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
)

month_index = (
    groupby_features['event_time_dt-first'] >>
    nvt.ops.LambdaOp(lambda col: ((col - col.min()) / np.timedelta64(1, 'M')).astype(int)) >>
    nvt.ops.Rename(f=lambda col: "month_index") >>
    nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
)

from nvtabular.ops import LambdaOp
import cudf

def map_day_index_vectorized(day_index_series):
    # Create an empty series with the same index as the input series to store the results
    result_series = cudf.Series(index=day_index_series.index, dtype="int")

    # Vectorized operations for each condition
    result_series[(1 <= day_index_series) & (day_index_series <= 90)] = 1
    result_series[(91 <= day_index_series) & (day_index_series <= 120)] = 2
    result_series[(121 <= day_index_series) & (day_index_series <= 160)] = 3
    result_series[(161 <= day_index_series) & (day_index_series <= 180)] = 4
    result_series[(181 <= day_index_series) & (day_index_series <= 210)] = 5
    result_series[(211 <= day_index_series) & (day_index_series <= 240)] = 6
    result_series[(241 <= day_index_series) & (day_index_series <= 1039)] = 7
    

    # Handle any values outside the specified ranges (if necessary)
    # result_series[day_index_series > 1039] = None

    return result_series

# Use the vectorized function with LambdaOp
month_index = (
    groupby_features['event_time_dt-first'] >>
    nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >>
    nvt.ops.LambdaOp(map_day_index_vectorized) >>
    nvt.ops.Rename(f=lambda col: "month_index") >>
    nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
)

"""

# tag session_id column for serving with legacy api
sess_id = groupby_features['session_id'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])

# Select features for training 
selected_features = sess_id + groupby_features['item_id-count'] + groupby_features_truncated + year_index

# Filter out sessions with less than 2 interactions 
MINIMUM_SESSION_LENGTH = 2
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] >= MINIMUM_SESSION_LENGTH) 
temp_workflow = nvt.Workflow(filtered_sessions)
temp_workflow.fit(nvt.Dataset(interactions_merged_df))
filtered_sessions_output = temp_workflow.transform(nvt.Dataset(interactions_merged_df)).to_ddf().compute()
#print("Filtered Sessions:")
#print(filtered_sessions_output.head(200))

dataset = nvt.Dataset(interactions_merged_df)
workflow = nvt.Workflow(filtered_sessions)

#print("length of dataset=====", dataset.size())
workflow.fit_transform(dataset).to_parquet(os.path.join(DATA_FOLDER, "book30daystimebased2_processed_nvt"))

workflow.output_schema

workflow.save(os.path.join(DATA_FOLDER, "book30daystimebased2_workflow_etl"))

sessions_gdf = cudf.read_parquet(os.path.join(DATA_FOLDER, "book30daystimebased2_processed_nvt/part_0.parquet"))


sessions_gdf = sessions_gdf[sessions_gdf.year_index>=1]

#print(sessions_gdf.head(3))

from transformers4rec.utils.data_utils import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir=os.path.join(DATA_FOLDER, "book30daystimebased2_preproc_sessions_by_year"),
                       partition_col='year_index',
                       timestamp_col='session_id',
                      )
del  sessions_gdf
gc.collect()

