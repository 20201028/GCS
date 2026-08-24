FIRE: Label-Free Generalized Attributed Community Search across Graphs

A PyTorch + torch-geometric implementation

Quick Start
Running cora
nohup python -u main.py --dataset cora --latent_steps 20 --attr 0 > resultOp20CA.out 2>&1 &

Key Parameters
All the parameters with their default value are in main.py

Project Structure
main.py         # begin here
data_load.py         # generate tasks for different dataset
QueryDataset.py  # extract query from subgraphs
train_eval.py                       # train, valid and test for IACS
Model.py                      # model for IACS
Layer.py                      # GATBias layer and FiLM layer
Loss.py

The Cora/Citeseer/Reddit datasets are from PyTorch_Geometric; The Facebook/Twitter datasets are from [SNAP] (https://snap.stanford.edu/data).

Contact
Open an issue or send email to shfang@se.cuhk.edu.hk if you have any problem
