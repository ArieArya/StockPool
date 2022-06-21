#######################
### Library Imports ###
#######################
# torch geometric
import torch
from torch import nn
from torch_geometric_temporal.signal import temporal_signal_split

# general libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm

# matplotlib
from matplotlib.pyplot import figure
plt.rcParams.update({'font.size': 16})
plt.style.use('seaborn-whitegrid')

# import local libraries
import GraphConstruction
from StockPool import StockPool, GNN

# set seed
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed)
  
# GPU compatibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define main function
if __name__ == "__main__":
    
    # extract all stock quotes
    snp = pd.read_csv('snp_info/snp_info.csv')
    stock_quotes = list(snp['Symbol'])
    quote_to_idx = {stock_quote: i for i, stock_quote in enumerate(stock_quotes)}
    idx_to_quote = {i: stock_quote for i, stock_quote in enumerate(stock_quotes)}

    # obtain node feature embeddings
    node_features = {}
    print('''
          
          1. Fetching Historical Data
          ''')
    for stock_quote in tqdm(stock_quotes):
        data = pd.read_csv('historical_data_snp/' + stock_quote + '_preprocessed.csv')
        node_features[stock_quote] = data

    # obtain quote to index and index to quote hash map
    quote_to_idx = {stock_quote: i for i, stock_quote in enumerate(stock_quotes)}
    idx_to_quote = {i: stock_quote for i, stock_quote in enumerate(stock_quotes)}

    
    # --------------------------------------------------
    # --- Obtain StockPool Cluster Assignment Matrix ---
    # --------------------------------------------------
    # obtain GICS sub-industries
    gics_subindustry = list(
        snp[snp['Symbol'].isin(stock_quotes)]['GICS Sub Industry'])

    # sort to unique index
    gics_subindustry_map = {}
    idx = 0
    for sub in gics_subindustry:
        if sub not in gics_subindustry_map:
            gics_subindustry_map[sub] = idx
            idx += 1

    # compute the GICS Subindustry matrix
    S_subindustry = torch.zeros(len(stock_quotes), len(gics_subindustry_map)).to(device)
    for i in range(len(stock_quotes)):
        subindustry = snp[snp['Symbol'] == stock_quotes[i]]['GICS Sub Industry'].item()
        S_subindustry[i, gics_subindustry_map[subindustry]] = 1

    gics_industry = snp[snp['Symbol'].isin(stock_quotes)]['GICS Sector']

    # sort to unique index
    gics_industry_map = {}
    idx = 0
    for industry in gics_industry:
        if industry not in gics_industry_map:
            gics_industry_map[industry] = idx
            idx += 1

    # compute the GICS industry matrix
    gics_subindustry_keys = list(gics_subindustry_map.keys())
    S_industry = torch.zeros(len(gics_subindustry_map), len(gics_industry_map)).to(device)
    for i in range(len(gics_subindustry_keys)):
        industry = snp[snp['GICS Sub Industry'] == gics_subindustry_keys[i]]['GICS Sector'].iloc[0]
    S_industry[i, gics_industry_map[industry]] = 1


    # -----------------------
    # --- Construct Graph ---
    # -----------------------
    print('''
          
          2. Constructing Stock Graph
          ''')
    periods = 10
    graph_construction = 'dtw'
    if graph_construction == 'dtw':
        # DTW graph
        dataset = GraphConstruction.build_dtw_graph_dataset(quote_to_idx, node_features, stock_quotes, lags=periods,
                                                        process_dtw=False, dtw_threshold=16, bin_out=True)
    elif graph_construction == 'corr':
        # correlation graph
        dataset = GraphConstruction.build_correlation_graph_dataset(quote_to_idx, node_features, stock_quotes, lags=periods,
                                                                    process_correlation=False, corr_threshold=0.6, bin_out=True)
    else:
        # sector-wise graph
        dataset = GraphConstruction.build_sector_graph_dataset(snp, quote_to_idx, node_features, stock_quotes, lags=periods, bin_out=True)
    
    # -----------------------
    # --- Preprocess Data ---
    # -----------------------
    # split to train and test set
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    # define num features and num nodes
    num_features = dataset[0].num_features
    num_nodes = dataset[0].num_nodes

    # define function to extract the static adjacency matrix
    def to_adj_mtx(data_point):
        adj_mtx_size = data_point.x.shape[0]
        adj_mtx = [[0.0] * adj_mtx_size for _ in range(adj_mtx_size)]
        edge_index = data_point.edge_index
        for i in range(len(edge_index[0])):
            adj_mtx[edge_index[0][i]][edge_index[1][i]] = 1.0
        return torch.tensor([adj_mtx]).to(device)
    adj_mtx = to_adj_mtx(dataset[0])

    # -------------------
    # --- Train Model ---
    # -------------------
    print('''
          
          3. Training / Loading Model
          ''')
    train_model = False
    model = StockPool(hidden_nodes=32, num_features=num_features, num_nodes=num_nodes,
                      S_subindustry=S_subindustry, S_industry=S_industry).to(device)

    # Load Existing Model
    if not train_model:
        model.load_state_dict(torch.load('saved_models/StockPool.pth'))
        print("Model successfully loaded...")

    # Train Model
    else:
        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

        # set model on training mode
        model.train()

        # define BCE loss
        bce_loss = nn.BCELoss()

        # train model
        epochs = 500
        patience = 200 # patience for early stopping
        best_cost = 100 # set to arbitrarily high value
        print("Training Model...")
        for epoch in range(epochs):
            cost = 0
            for time, snapshot in enumerate(train_dataset):
                snapshot = snapshot.to(device)
                y_hat = model(snapshot.x, snapshot.edge_index, adj_mtx, snapshot.edge_attr)
                cost = cost + bce_loss(y_hat.flatten().to(torch.float32), snapshot.y.to(torch.float32)) 
            cost = cost / (time+1)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

            # evaluate on test set
            correct_preds = 0
            total_preds = 0
            test_cost = 0
            for time, snapshot in enumerate(test_dataset):
                snapshot = snapshot.to(device)
                y_hat = model(snapshot.x, snapshot.edge_index, adj_mtx, snapshot.edge_attr)
                test_cost = test_cost + bce_loss(y_hat.flatten().to(torch.float32), snapshot.y.to(torch.float32)) 

                # evaluate test accuracy
                for i, pred in enumerate(y_hat):
                    # check if predicted returns and true returns have the same sign
                    if round(pred.item()) == round(snapshot.y[i].item()):
                        correct_preds += 1
                    total_preds += 1
            test_acc = correct_preds / total_preds
            test_cost = test_cost / (time + 1)

            # Early Stopping
            if test_cost < best_cost:
                best_cost = test_cost
                patience = max(patience, 50)
            else:
                patience -= 1
                if patience == 0:
                    print("early stopping triggered...")
                    # break

            print("Epoch {}/{} Train BCE: {:.6f}, Test BCE: {:.6f}, Test Accuracy: {:.6f}".format(epoch, epochs, cost.item(), test_cost.item(), test_acc))
        
    # ----------------------
    # --- Evaluate Model ---
    # ----------------------
    print('''
          
          4. Evaluating Model - Binary Accuracy
          ''')
    model.eval()
    correct_preds = 0
    base_correct_preds = 0
    total_preds = 0
    for time, snapshot in enumerate(test_dataset):
        snapshot = snapshot.to(device)
        y_hat = model(snapshot.x, snapshot.edge_index, adj_mtx, snapshot.edge_attr)
        for i, pred in enumerate(y_hat):
            # check if predicted returns and true returns have the same sign
            if round(pred.item()) == round(snapshot.y[i].item()):
                correct_preds += 1
            if round(snapshot.y[i].item()) == 1:
                base_correct_preds += 1
            total_preds += 1

    print("Base Binary Accuracy: {:.20f}".format(base_correct_preds/total_preds))
    print("StockPool GNN Binary Accuracy: {:.20f}".format(correct_preds/total_preds))
    
    # -------------------------------------------
    # --- Obtain Binary Accuracy Per Industry ---
    # -------------------------------------------
    company_bin_acc = {}

    # evaluate model binary accuracy using test set
    for time, snapshot in enumerate(test_dataset):
        snapshot = snapshot.to(device)
        y_hat = model(snapshot.x, snapshot.edge_index, adj_mtx, snapshot.edge_attr)
        total_preds = len(y_hat)
        for i, pred in enumerate(y_hat):
          cur_quote = idx_to_quote[i]

          # add quote to dictionary
          if cur_quote not in company_bin_acc:
            company_bin_acc[cur_quote] = {'correct_preds': 0, 'base_correct_preds': 0}

          # check if predicted returns and true returns have the same sign
          if round(pred.item()) == round(snapshot.y[i].item()):
            company_bin_acc[cur_quote]['correct_preds'] += 1
          if round(snapshot.y[i].item()) == 1:
            company_bin_acc[cur_quote]['base_correct_preds'] += 1

    # compute accuracy per stock
    for quote in company_bin_acc:
      company_bin_acc[quote]['correct_preds'] /= time
      company_bin_acc[quote]['base_correct_preds'] /= time
    
    # compute accuracy per industry
    industry_bin_acc = {}
    for quote in company_bin_acc:
      cur_industry = snp[snp['Symbol'] == quote]['GICS Sector'].item()
      if cur_industry not in industry_bin_acc:
        industry_bin_acc[cur_industry] = {'acc': 0, 'base_acc': 0, 'num_companies': 0}
      industry_bin_acc[cur_industry]['acc'] += company_bin_acc[quote]['correct_preds']
      industry_bin_acc[cur_industry]['base_acc'] += company_bin_acc[quote]['base_correct_preds']
      industry_bin_acc[cur_industry]['num_companies'] += 1

    for industry in industry_bin_acc:
      industry_bin_acc[industry]['acc'] /= industry_bin_acc[industry]['num_companies']
      industry_bin_acc[industry]['base_acc'] /= industry_bin_acc[industry]['num_companies']

    # construct result table
    bin_acc_table = {}
    for industry in industry_bin_acc:
      bin_acc_table[industry] = [industry_bin_acc[industry]['base_acc'], industry_bin_acc[industry]['acc']]
    bin_acc_table = pd.DataFrame.from_dict(bin_acc_table, orient='index',
                           columns=['Base Accuracy', 'StockPool GNN Accuracy'])
    print(bin_acc_table)


    # ---------------------------
    # --- Perform Backtesting ---
    # ---------------------------
    print('''
          
          5. Perform Backtesting Simulation
          ''')
    
    # rebuild a new DTW graph dataset with log-returns output
    logret_dataset = GraphConstruction.build_dtw_graph_dataset(quote_to_idx, node_features, stock_quotes, lags=periods,
                                                                         process_dtw=False, dtw_threshold=16, bin_out=False)
    _, test_logret_dataset = temporal_signal_split(logret_dataset, train_ratio=0.8)
    
    # obtain base cumulative returns
    base_backtest_ret = []
    for time, snapshot in enumerate(test_dataset):
        snapshot = snapshot.to(device)
        base_backtest_ret.append([])

        # obtain true log-returns
        cur_logret = test_logret_dataset[time].y

        # iterate over each stock
        for i in range(len(cur_logret)):
            # add to prediction
            base_backtest_ret[-1].append(cur_logret[i].item())

    # perform backtesting on model
    backtest_ret = []
    for time, snapshot in enumerate(test_dataset):
        snapshot = snapshot.to(device)
        backtest_ret.append([])

        # obtain prediction
        y_hat = model(snapshot.x, snapshot.edge_index, adj_mtx, snapshot.edge_attr)

        # obtain true log-returns
        cur_logret = test_logret_dataset[time].y

        # iterate over each stock
        for i in range(len(y_hat)):
            # obtain current stock prediction
            cur_pred = round(y_hat[i].item())

            # if prediction is up, put current log-return
            if cur_pred == 1:
                pred_logret = cur_logret[i].item()
            # if prediction is down, reverse (minus) current log-return (i.e. we are shorting)
            else:
                pred_logret = -1 * cur_logret[i].item()

            # add to prediction
            backtest_ret[-1].append(pred_logret)

    # obtain mean log-returns
    base_backtest_mean_ret = [sum(x)/len(x) for x in base_backtest_ret]
    backtest_mean_ret = [sum(x)/len(x) for x in backtest_ret]

    # obtain the base S&P500 Market Sharpe Ratio
    mean_daily_ret = np.mean([math.exp(sum(x)/len(x))-1 for x in base_backtest_ret])
    std_daily_ret = np.std([math.exp(sum(x)/len(x))-1 for x in base_backtest_ret])
    risk_free_rate = (3.03/100)/365
    sharpe_ratio = (mean_daily_ret - risk_free_rate) / \
        std_daily_ret * math.sqrt(252)
    print("S&P500 Sharpe Ratio: ", sharpe_ratio)

    # obtain the model Sharpe Ratio
    mean_daily_ret = np.mean([math.exp(sum(x)/len(x))-1 for x in backtest_ret])
    std_daily_ret = np.std([math.exp(sum(x)/len(x))-1 for x in backtest_ret])
    risk_free_rate = (3.03/100)/365
    sharpe_ratio = (mean_daily_ret - risk_free_rate) / \
        std_daily_ret * math.sqrt(252)
    print("StockPool GNN Sharpe Ratio: ", sharpe_ratio)

    # obtain cumulative returns
    base_backtest_mean_cumret = [math.exp(x)for x in np.cumsum(base_backtest_mean_ret)]
    backtest_mean_cumret = [math.exp(x) for x in np.cumsum(backtest_mean_ret)]

    # plot results
    plt.figure(figsize=(16, 7), dpi=100)
    plt.plot([x for x in range(len(base_backtest_mean_cumret))], base_backtest_mean_cumret, label='S&P500 Base Cumulative Returns', linestyle='--')
    plt.plot([x for x in range(len(backtest_mean_cumret))], backtest_mean_cumret, label='StockPool GNN Cumulative Returns')
    plt.ylabel('Cumulative Returns')
    plt.xlabel('Days')
    plt.legend()
    plt.show()
