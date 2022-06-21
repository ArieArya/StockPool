import pandas as pd
import pickle
import numpy as np
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from tqdm import tqdm

#########################
### Correlation Graph ###
#########################
def build_correlation_graph_dataset(quote_to_idx, features_df_dict, stock_quotes, lags=12, process_correlation=False, corr_threshold=0.6, bin_out=True):
  # obtain edge correlation -> convert to dictionary for O(1) search complexity
  correlation_dict = {}
  if process_correlation:
    correlation_data = pd.read_csv('snp_info/cross_correlation_snp.csv')
    for index, row in correlation_data.iterrows():
      correlation_dict[f'{row["Symbol_1"]}_{row["Symbol_2"]}'] = row['Correlation']
    # save correlation data
    with open('snp_info/snp_correlation_dict.pkl', 'wb') as f:
      pickle.dump(correlation_dict, f)
  else:
    # load correlation data
    with open('snp_info/snp_correlation_dict.pkl', 'rb') as f:
      correlation_dict = pickle.load(f)

  data_len = len(features_df_dict[stock_quotes[1]])

  # define required parameters
  features = []
  targets = []
  edge_index = [[], []]
  edge_weight = []

  # obtain static graph connections
  print("Constructing correlation graph...")
  for i, stock_quote_1 in tqdm(enumerate(stock_quotes)):
    idx_1 = quote_to_idx[stock_quote_1]
    for j in range(i+1, len(stock_quotes)):
      stock_quote_2 = stock_quotes[j]
      idx_2 = quote_to_idx[stock_quote_2]
      if stock_quote_1 != stock_quote_2:
        try:
          # check if correlation higher than threhsold
          try:
            corr = correlation_dict[f'{stock_quote_1}_{stock_quote_2}']
          except:
            corr = correlation_dict[f'{stock_quote_2}_{stock_quote_1}']

          if abs(corr) > corr_threshold:
            # set connectivity attribute as correlation value or fixed
            edge_weight.append(corr)
            edge_weight.append(corr)

            # create undirected connectivity
            edge_index[0].append(idx_1)
            edge_index[0].append(idx_2)
            edge_index[1].append(idx_2)
            edge_index[1].append(idx_1)

        except Exception as e:
          print(e)

  print("Building node feature embeddings...")
  # obtain node features at each snapshot
  for i in tqdm(range(data_len - lags + 1)):
    f1 = []
    for stock_quote in stock_quotes:
      f2 = []
      try:
        if f2:
          f2 = f2[1:]
          f2.append(features_df_dict[stock_quote]
                    ['log_returns_norm'][i + lags - 1])
        else:
          for j in range(lags):
            idx = quote_to_idx[stock_quote]
            f2.append(features_df_dict[stock_quote]['log_returns_norm'][i + j])
        f1.append(f2)
      except Exception as e:
        print(e)
    features.append(f1)

  # obtain target at each snapshot (either binary output or next-day return)
  for i in range(lags-1, data_len):
    target = []
    for stock_quote in stock_quotes:
        if bin_out:
            target.append(features_df_dict[stock_quote]['bin_output'][i])
        else:
            target.append(features_df_dict[stock_quote]['out_log_returns'][i])
    targets.append(target)

  # convert to torch tensor
  features = np.array(features)
  targets = np.array(targets)
  edge_index = np.array(edge_index)
  edge_weight = np.array(edge_weight)

  # create static graph temporal signal dataset
  dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight,
                                      features=features, targets=targets)
  return dataset


#########################
### Sector-Wise Graph ###
#########################
def build_sector_graph_dataset(snp, quote_to_idx, features_df_dict, stock_quotes, lags=12, bin_out=True):
  # construct sector dictionary
  sector_dict = {}
  for idx, row in snp.iterrows():
      sector_dict[row['Symbol']] = row['GICS Sector']

  data_len = len(features_df_dict[stock_quotes[1]])

  # define required parameters
  features = []
  targets = []
  edge_index = [[], []]
  edge_weight = []

  # obtain static graph connections
  print("Constructing sector-wise graph...")
  for i, stock_quote_1 in tqdm(enumerate(stock_quotes)):
    idx_1 = quote_to_idx[stock_quote_1]
    for j in range(i+1, len(stock_quotes)):
      stock_quote_2 = stock_quotes[j]
      idx_2 = quote_to_idx[stock_quote_2]
      if stock_quote_1 != stock_quote_2:
        if sector_dict[stock_quote_1] == sector_dict[stock_quote_2]:
          # set connectivity attribute as 1
          edge_weight.append(1)
          edge_weight.append(1)

          # set undirected connectivity
          edge_index[0].append(idx_1)
          edge_index[0].append(idx_2)
          edge_index[1].append(idx_2)
          edge_index[1].append(idx_1)

  # obtain node features at each snapshot
  print("Building node feature embeddings...")
  for i in tqdm(range(data_len - lags + 1)):
    f1 = []
    for stock_quote in stock_quotes:
      f2 = []
      if f2:
        f2 = f2[1:]
        f2.append(features_df_dict[stock_quote]
                  ['log_returns_norm'][i + lags - 1])
      else:
        for j in range(lags):
          idx = quote_to_idx[stock_quote]
          f2.append(features_df_dict[stock_quote]['log_returns_norm'][i + j])
      f1.append(f2)
    features.append(f1)

  # obtain target at each snapshot
  for i in range(lags-1, data_len):
    target = []
    for stock_quote in stock_quotes:
        # change back to out_log_returns after
        if bin_out:
            target.append(features_df_dict[stock_quote]['bin_output'][i])
        else:
            target.append(features_df_dict[stock_quote]['out_log_returns'][i])
    targets.append(target)

  # convert to torch tensor
  features = np.array(features)
  targets = np.array(targets)
  edge_index = np.array(edge_index)
  edge_weight = np.array(edge_weight)

  # create static graph temporal signal dataset
  dataset = StaticGraphTemporalSignal(
      edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
  return dataset


#################
### DTW Graph ###
#################
# builds static DTW temporal graph dataset
def build_dtw_graph_dataset(quote_to_idx, features_df_dict, stock_quotes, lags=12, process_dtw=False, dtw_threshold=16, bin_out=True):
  # obtain edge DTW -> convert to dictionary for O(1) search complexity
  dtw_dict = {}
  if process_dtw:
    dtw_data = pd.read_csv('snp_info/dtw_snp.csv')
    for index, row in dtw_data.iterrows():
      dtw_dict[f'{row["Symbol_1"]}_{row["Symbol_2"]}'] = row['DTW']
    # save dtw data
    with open('snp_info/snp_dtw_dict.pkl', 'wb') as f:
      pickle.dump(dtw_dict, f)
  else:
    # load correlation data
    with open('snp_info/snp_dtw_dict.pkl', 'rb') as f:
      dtw_dict = pickle.load(f)

  data_len = len(features_df_dict[stock_quotes[1]])

  # define required parameters
  features = []
  targets = []
  edge_index = [[], []]
  edge_weight = []

  # obtain static graph connections
  print("Constructing DTW graph...")
  for i, stock_quote_1 in tqdm(enumerate(stock_quotes)):
    idx_1 = quote_to_idx[stock_quote_1]
    for j in range(i+1, len(stock_quotes)):
      stock_quote_2 = stock_quotes[j]
      idx_2 = quote_to_idx[stock_quote_2]
      if stock_quote_1 != stock_quote_2:
        try:
          # check if correlation higher than threhsold
          try:
            dtw_dist = dtw_dict[f'{stock_quote_1}_{stock_quote_2}']
          except:
            dtw_dist = dtw_dict[f'{stock_quote_2}_{stock_quote_1}']

          if abs(dtw_dist) < dtw_threshold:
            # set connectivity attribute as dtw value or fixed
            edge_weight.append(1)
            edge_weight.append(1)

            # create undirected connectivity
            edge_index[0].append(idx_1)
            edge_index[0].append(idx_2)
            edge_index[1].append(idx_2)
            edge_index[1].append(idx_1)

        except Exception as e:
          print(e)

  # obtain node features at each snapshot
  print("Building node feature embeddings...")
  for i in tqdm(range(data_len - lags + 1)):
    f1 = []
    for stock_quote in stock_quotes:
      f2 = []
      try:
        if f2:
          f2 = f2[1:]
          f2.append(features_df_dict[stock_quote]
                    ['log_returns_norm'][i + lags - 1])
        else:
          for j in range(lags):
            idx = quote_to_idx[stock_quote]
            f2.append(features_df_dict[stock_quote]['log_returns_norm'][i + j])
        f1.append(f2)
      except Exception as e:
        print(e)
    features.append(f1)

  # obtain target at each snapshot (either binary output or next-day return)
  for i in range(lags-1, data_len):
    target = []
    for stock_quote in stock_quotes:
        if bin_out:
            target.append(features_df_dict[stock_quote]['bin_output'][i])
        else:
            target.append(features_df_dict[stock_quote]['out_log_returns'][i])
    targets.append(target)

  # convert to torch tensor
  features = np.array(features)
  targets = np.array(targets)
  edge_index = np.array(edge_index)
  edge_weight = np.array(edge_weight)

  # create static graph temporal signal dataset
  dataset = StaticGraphTemporalSignal(
      edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
  return dataset
