# main_ssl.py — Part B runner (SSL)
from Networks.ssl_model import *
import argparse, yaml, os
from os.path import dirname, abspath

rootDirectory = dirname(abspath(__file__))
datasetDirectory = os.path.join(rootDirectory, "Dataset")
imgDirectory     = os.path.join(datasetDirectory, "images")
maskDirectory    = os.path.join(datasetDirectory, "annotations")

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='BestConfigB')

def main(args):
    yaml_path = os.path.join(rootDirectory, "Todo_List", args.exp + ".yaml")
    with open(yaml_path, "r") as f:
        param = yaml.safe_load(f)

    resultsPath = os.path.join(rootDirectory, "Results", args.exp)
    myNetwork = Network_Class(param, imgDirectory, maskDirectory, resultsPath)

    print("Start SSL training (Part B)")
    myNetwork.train()
    print("SSL training done")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
