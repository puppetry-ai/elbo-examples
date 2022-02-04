import torch
import numpy as np
from pytorch_fid import fid_score
import scipy

# Based on https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def calculate_fid(validation_samples, generated_samples):
    # calculate mean and covariance statistics
    val = torch.cat(validation_samples, 2).squeeze(0)
    gen = torch.cat(generated_samples, 1).squeeze(0).T

    mu1, sigma1 = torch.mean(val, dim=1), torch.cov(val)
    mu2, sigma2 = torch.mean(gen, dim=1), torch.cov(gen)

    mu1 = mu1.cpu().detach().numpy()
    mu2 = mu2.cpu().detach().numpy()
    sigma1 = sigma1.cpu().detach().numpy()
    sigma2 = sigma2.cpu().detach().numpy()

    fid = fid_score.calculate_frechet_distance(mu2, sigma2, mu1, sigma1)
    return fid
