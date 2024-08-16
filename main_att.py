import argparse
import os
import numpy as np
from tqdm import tqdm
import attention_corr_methods_adapted as adacorr
import torch 

def load_representations_and_neurons(models):
    """
    Create representations and neurons dictionaries based on the given models.
    """
    representations = {}
    neurons = {}

    for model in models:
        if 'base' in model:
            for i in range(12):
                representations[f'{model}_{i}_att'] = f'{model}_{i}_att'
                neurons[f'{model}_{i}_att'] = 12
        else:
            for i in range(24):
                representations[f'{model}_{i}_att'] = f'{model}_{i}_att'
                neurons[f'{model}_{i}_att'] = 16

    return representations, neurons

def create_all_pairs(representations):
    """
    Create all unique pairs of representations.
    """
    all_pairs = []

    for network, other_network in tqdm([(net1, net2) for net1 in representations for net2 in representations], desc='correlate'):
        all_pairs.append((network, other_network))
    
    return all_pairs

def main(args):
    models = args.models.split(',')

    representations, neurons = load_representations_and_neurons(models)
    all_pairs = create_all_pairs(representations)

    # Process each pair and compute correlations
    for pair in all_pairs:
        
        network, other_network = pair
        representations = {network:[], other_network:[]}
        loads = {
                    network: np.load(f'XSpeech_SSL/outputs/embds/{network}.npz'),
                    other_network: np.load(f'XSpeech_SSL/outputs/embds/{other_network}.npz')
                }
        
        representations[network].extend(map(torch.tensor, loads[network].values()))
        representations[other_network].extend(map(torch.tensor, loads[other_network].values()))
        neurons = {network:12 if 'base' in network else 16, other_network:12 if 'base' in other_network else 16}
        
        ## FroMaxMinCorr
        corr = adacorr.FroMaxMinCorr(neurons, representations, op=max, device='cuda')
        corr.compute_correlations()
        corr.write_correlations(f'{args.output_dir}/FroMaxMinCorr_{network}_{other_network}_att.pkl')
        
        ## PearsonMaxMinCorr
        corr = adacorr.PearsonMaxMinCorr(neurons, representations, op=max, device='cuda')
        corr.compute_correlations()
        corr.write_correlations(f'{args.output_dir}/PearsonMaxMinCorr_{network}_{other_network}_att.pkl')

        ## JSMaxMinCorr
        corr = adacorr.JSMaxMinCorr(neurons, representations, op=max, device='cuda')
        corr.compute_correlations()
        corr.write_correlations(f'{args.output_dir}/JSMaxMinCorr_{network}_{other_network}_att.pkl')


        network, other_network = pair
        representations = {network:network, other_network:other_network}
        neurons = {network:768 if 'base' in network else 1024, other_network:768 if 'base' in other_network else 1024}
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlations between different network representations.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output correlation matrices.")
    parser.add_argument('--models', type=str, required=True, help="Comma-separated list of model names (e.g., 'hub_base,hub_large,w2v_base,w2v_large').")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/matrix_output', exist_ok=True)

    main(args)
