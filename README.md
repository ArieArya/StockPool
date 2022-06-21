# Financial Modelling via Graph Neural Networks with Custom Pooling
This repository contains only the core code and S&P500 dataset for the implementation of the StockPool GNN model. The full original codebase is based on Google Colaboratory, 
and is not fully ported into this online repository.

---

### PyTorch Dependencies
```
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-geometric-temporal -f https://data.pyg.org/whl/torch-1.11.0%2Bcpu.html
pip3 install -q pthflops
```

---

### General File Description
- **main.py** contains the Python script to setup, train, and evaluate the StockPool GNN model
- **StockPool.py** contains the StockPool GNN model
- **GraphConstruction.py** contains the three methods of stock graph construction (sector-wise, correlation, DTW)
- *./historical_data_snp/* contains the preprocessed historical S&P500 data
- *./saved_models/* contains a saved StockPool GNN model whose performance is quoted in the dissertation paper
- *./snp_info/* contains core information regarding the S&P500 dataset, including the GICS industries & subindustries, cross-correlation, and DTW distance

---

### How to Run
The **main.py** script will run all the required steps, including building the stock graph, initializing the model, training the model, evaluating the model,
and performing the backtesting simulation. Note that changing different parameters must be done from **main.py**. The default setting includes the use of
DTW graph, loading instead of training a new model, etc.

```
python main.py
```

