import torch
import os
import argparse
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import remove_dim, save_output_temp, change_device, read_feature, ImageDataset, get_matches
from assets.lightglue import LightGlue
from assets.superpoint import SuperPoint

def extract_features(index, dataset, extractor, device, temp_path):
    """Extracts and saves features for a given image index."""
    image_tensor = dataset[index].to(device)
    with torch.no_grad():
        output = extractor.extract(image_tensor)
    save_output_temp(output, temp_path, str(index))

def compute_similarity(pair):
    """Computes similarity between two images using pre-extracted features."""
    idx, jdx, temp_path, device = pair
    feat0 = read_feature(temp_path, str(idx), device)
    feat1 = read_feature(temp_path, str(jdx), device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    with torch.no_grad():
        matches01 = matcher({"image0": feat0, "image1": feat1})
    
    output = remove_dim([feat0, feat1, matches01])
    output = [change_device(t, 'cpu') for t in output]
    return (idx, jdx, get_matches(output, True))

def run(response_path: str, output_path: str) -> None:
    dataset = ImageDataset(response_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # Reduce keypoints for speed
    
    temp_path = os.path.join(output_path, 'temp_features')
    os.makedirs(temp_path, exist_ok=True)
    
    # Step 1: Extract features for all images
    for idx in tqdm(range(len(dataset)), desc="Extracting Features"):
        extract_features(idx, dataset, extractor, device, temp_path)
    
    # Step 2: Compute similarities in parallel
    pairs = [(idx, jdx, temp_path, device) for idx in range(len(dataset)) for jdx in range(len(dataset))]
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_similarity, pairs), total=len(pairs), desc="Computing Similarities"))
    
    # Step 3: Save results to JSON
    save_dict = {f"({dataset.image_paths[idx]}, {dataset.image_paths[jdx]})": score for idx, jdx, score in results}
    
    with open(os.path.join(output_path, "output.json"), 'w') as f:
        json.dump(save_dict, f)

    shutil.rmtree(temp_path, ignore_errors=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_path', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output JSON and histogram')
    args = parser.parse_args()
    run(args.response_path, args.output_path)


