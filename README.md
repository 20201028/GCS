FIRE: Label-Free Generalized Attributed Community Search across Graphs

A PyTorch + torch-geometric implementation

Quick Start
Running cora
nohup python -u main.py --dataset cora --latent_steps 20 --attr 0 > resultOp20CA.out 2>&1 &

Key Parameters
All the parameters with their default value are in config.py

Project Structure
main.py         # begin here
sampleQuery.py         # generate query
models                      # model for FIRE
DataSet                      

The Cora/Citeseer datasets are from [linqs](https://linqs.org/datasets); The Facebook/Twitter datasets are from [SNAP] (https://snap.stanford.edu/data).

Contact
Open an issue or send email to sunch23@stumail.neu.edu.cn if you have any problem
