# main.py
from Networks.model import *
from Dataset.makeGraph import *

import argparse
import yaml
import os
from os.path import dirname, abspath

rootDirectory    = dirname(abspath(__file__))
datasetDirectory = os.path.join(rootDirectory, "Dataset")
imgDirectory     = os.path.join(datasetDirectory, "images")
maskDirectory    = os.path.join(datasetDirectory, "annotations")

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='DefaultExp')

def main(args):
    # --- 0) read yaml config (absolute path)
    yaml_path = os.path.join(rootDirectory, "Todo_List", f"{args.exp}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        param = yaml.safe_load(f)

    resultsPath = os.path.join(rootDirectory, "Results", args.exp)

    # --- 1) network
    myNetwork = Network_Class(param, imgDirectory, maskDirectory, resultsPath)

    # --- 2) dataset viz optional (controlled by YAML)
    if bool(param.get("SHOW_DATASET", True)):
        try:
            showDataset(myNetwork.dataSetTrain, param)
        except Exception as e:
            print(f"[WARN] showDataset skipped: {e}")

    # --- 3) train
    print("Start to train the network")
    myNetwork.train()
    print("The network is trained")

    # --- 4) eval
    myNetwork.loadWeights()
    myNetwork.evaluate()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
