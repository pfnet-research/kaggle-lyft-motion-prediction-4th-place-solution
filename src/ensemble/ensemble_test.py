import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from numpy.random import multinomial, multivariate_normal
from sklearn.mixture import GaussianMixture
import warnings
warnings.simplefilter('ignore')

sys.path.append("./src")
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.mixture.gmm import GaussianMixtureIdentity

NUM_TEST = 71122
SEED = 0
np.random.seed(SEED)


def main(flags):
    outdir = flags["outdir"]
    w = flags["weight"]
    sigma = flags["sigma"]
    N_sample = flags["N_sample"]
    covariance_type = flags["covariance_type"]
    file_list = flags["file_list"]
    df_list = [pd.read_csv(file) for file in file_list]
    I = np.eye(100)*sigma
    samples = [multivariate_normal(np.zeros(100), I) for _ in range(N_sample)]
    data = []

    for idx in tqdm(range(NUM_TEST)):
        rows =  [df_list[i].loc[idx] for i in range(len(df_list))]
        mu = [np.vstack([rows[i][5:105].values,rows[i][105:205].values,rows[i][205:305].values])
              for i in range(len(rows))]
        mu = np.concatenate(mu)
        confidence = [w[i]*rows[i][2:5].values for i in range(len(w))]
        confidence = np.concatenate(confidence)
        confidence /= confidence.sum()
        x = mu[np.random.choice(3*len(w), size=N_sample, p=confidence)]+samples
        if covariance_type=="identity":
            gauss = GaussianMixtureIdentity(3, "spherical", random_state=SEED)
        else:
            gauss = GaussianMixture(3, covariance_type, random_state=SEED)
        gauss.fit(x)
        confidence_fit = gauss.weights_
        mu_fit = gauss.means_
        row = [None, None]
        row += confidence_fit.tolist()
        row += mu_fit.reshape(-1).tolist()
        data.append(row)


    df = pd.DataFrame(data=data,columns=df_list[0].columns)
    df["timestamp"] = df_list[0]["timestamp"]
    df["track_id"] = df_list[0]["track_id"]
    df.to_csv(outdir,index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--yaml_filepath', '-y', type=str,
                        help='Flags yaml file path')
    args = parser.parse_args()
    with open(args.yaml_filepath, 'r') as f:
        flags = yaml.safe_load(f)
    main(flags)
