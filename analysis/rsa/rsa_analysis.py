"""Representational Similarity Analysis (RSA) functions for comparing neural representations."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def compute_rdm(embeddings, metric='correlation'):
    """
    Compute Representational Dissimilarity Matrix (RDM).
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        metric: distance metric ('correlation', 'euclidean', 'cosine')
    
    Returns:
        RDM: numpy array of shape (n_samples, n_samples)
    """
    if metric == 'correlation':
        # Pearson correlation distance: 1 - correlation
        rdm = 1 - np.corrcoef(embeddings)
    else:
        # Use scipy's pdist for other metrics
        rdm = squareform(pdist(embeddings, metric=metric))
    
    return rdm


def compare_rdms(rdm1, rdm2, method='spearman'):
    """
    Compare two RDMs using correlation.
    
    Args:
        rdm1, rdm2: RDMs to compare
        method: 'spearman' or 'pearson'
    
    Returns:
        correlation coefficient and p-value
    """
    # Extract upper triangular part (excluding diagonal)
    upper_indices = np.triu_indices_from(rdm1, k=1)
    rdm1_upper = rdm1[upper_indices]
    rdm2_upper = rdm2[upper_indices]
    
    if method == 'spearman':
        corr, p_val = spearmanr(rdm1_upper, rdm2_upper)
    else:
        corr, p_val = pearsonr(rdm1_upper, rdm2_upper)
    
    return corr, p_val


def compute_rsa_scores(embeddings_dict, metric='correlation', comparison='spearman'):
    """
    Compute RSA scores between all pairs of embeddings.
    
    Args:
        embeddings_dict: dict with keys as names and values as embeddings
        metric: distance metric for RDM computation
        comparison: method for comparing RDMs
    
    Returns:
        dict of RSA scores between all pairs
    """
    # Compute RDMs for all embeddings
    rdms = {}
    for name, emb in embeddings_dict.items():
        rdms[name] = compute_rdm(emb, metric=metric)
    
    # Compare all pairs
    rsa_scores = {}
    names = list(rdms.keys())
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names[i+1:], i+1):
            corr, p_val = compare_rdms(rdms[name1], rdms[name2], method=comparison)
            rsa_scores[f"{name1}_vs_{name2}"] = {
                'correlation': corr,
                'p_value': p_val
            }
    
    return rsa_scores, rdms


def plot_rdm(rdm, title='RDM', cmap='viridis', figsize=(8, 6)):
    """
    Plot Representational Dissimilarity Matrix as heatmap.
    
    Args:
        rdm: RDM to plot
        title: plot title
        cmap: colormap
        figsize: figure size
    
    Returns:
        fig, ax: matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(rdm, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Dissimilarity', rotation=270, labelpad=20)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')
    
    return fig, ax


def plot_rdm_comparison(rdm1, rdm2, title1='RDM 1', title2='RDM 2', cmap='viridis'):
    """
    Plot two RDMs side by side for comparison.
    
    Args:
        rdm1, rdm2: RDMs to compare
        title1, title2: titles for subplots
        cmap: colormap
    
    Returns:
        fig: matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot first RDM
    im1 = ax1.imshow(rdm1, cmap=cmap, aspect='auto')
    ax1.set_title(title1)
    ax1.set_xlabel('Sample index')
    ax1.set_ylabel('Sample index')
    plt.colorbar(im1, ax=ax1, label='Dissimilarity')
    
    # Plot second RDM
    im2 = ax2.imshow(rdm2, cmap=cmap, aspect='auto')
    ax2.set_title(title2)
    ax2.set_xlabel('Sample index')
    ax2.set_ylabel('Sample index')
    plt.colorbar(im2, ax=ax2, label='Dissimilarity')
    
    plt.tight_layout()
    return fig


def compute_rdm_reliability(embeddings, n_splits=2, metric='correlation'):
    """
    Compute split-half reliability of RDM.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        n_splits: number of random splits to average
        metric: distance metric for RDM
    
    Returns:
        reliability score (correlation between split halves)
    """
    n_samples = embeddings.shape[0]
    reliabilities = []
    
    for _ in range(n_splits):
        # Random split
        indices = np.random.permutation(n_samples)
        split1 = indices[:n_samples//2]
        split2 = indices[n_samples//2:]
        
        # Compute RDMs for each split
        rdm1 = compute_rdm(embeddings[split1], metric=metric)
        rdm2 = compute_rdm(embeddings[split2], metric=metric)
        
        # Compare RDMs
        corr, _ = compare_rdms(rdm1, rdm2, method='spearman')
        reliabilities.append(corr)
    
    return np.mean(reliabilities)