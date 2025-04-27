import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from new_train import RFEncoder, RFPredictor, pool_mask, load_walkable_nodes

def main():
    df = pd.read_csv('train_data/training_walks.csv')
    df_tx2 = df[df['transmitter'] == 'tx2']

    



if __name__=='__main__':
    main()
