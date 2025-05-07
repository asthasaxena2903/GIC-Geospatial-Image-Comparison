
from utils import remove_dim, save_output_temp,save_output_macthes, change_device, read_feature, ImageDataset, get_neighbours, get_matches
import torch
import shutil
import os
import argparse
from assets.lightglue import LightGlue
from assets.superpoint import SuperPoint

from tqdm import tqdm
import json


def run(source_dir: str, output_dir: str, save_score: bool) -> None:
  """
    Extract and match keypoints from images in the given source directory, and save the results to an output directory.

    Args:
        source_dir (str): The path to the directory containing the input images.
        output_dir (str): The path to the directory where the output files will be saved.
        save_score (bool): Whether to include the match scores in the output or not.

    Steps:
        1. Load the image pairs and filenames using the `get_neighbours` function.
        2. Set up the device (CPU or GPU) for PyTorch operations.
        3. Initialize the SuperPoint keypoint extractor and LightGlue matcher.
        4. Create a temporary directory for storing feature files.
        5. For each image in the dataset:
            a. Extract keypoints and features using the SuperPoint extractor.
            b. Save the features to a temporary file.
        6. For each pair of images:
            a. Load the features from the temporary files.
            b. Match the features using the LightGlue matcher.
            c. Remove the batch dimension from the output tensors.
            d. Move the output tensors to CPU.
            e. Extract the matches using the `get_matches` function.
            f. Save the matches in a dictionary with the image pair as the key.
        7. Save the matches dictionary as a JSON file in the output directory.
        8. Remove the temporary directory.

    Returns:
        None
  """

  image_pairs, files = get_neighbours(source_dir)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
  matcher = LightGlue(features="superpoint").eval().to(device)

  dataset = ImageDataset(source_dir)
  temp_dir = os.path.join(output_dir, 'temp')
  os.makedirs(temp_dir)

  for i in tqdm(range(len(dataset)), total = len(dataset), disable = False, unit = 'Images', desc = 'Progress bar for inferencing superpoint model'):
      image = dataset[i].to(device)
      with torch.no_grad():
        output_dict = extractor.extract(image)
      save_output_temp(output_dict, temp_dir, str(i))
  
  save_dict = {}
  for image1, image2 in tqdm(image_pairs, total = len(image_pairs), disable = False, unit = 'image_pairs', desc = 'Progress bar for inferencing lightglue model'):

      feat0 = read_feature(temp_dir, str(image1), device)
      feat1 = read_feature(temp_dir, str(image2), device)

      with torch.no_grad():
        matches01 = matcher({"image0": feat0, "image1": feat1})
      output = remove_dim([feat0, feat1, matches01])
      for i in range(len(output)):
        output[i] = change_device(output[i], 'cpu')
      match_pair = get_matches(output, save_score)
      image_pair = f"({files[image1]}, {files[image2]})"
      save_dict[image_pair] = match_pair

  json_path = os.path.join(output_dir, "output.json")
  with open(json_path, 'w') as f:
    json.dump(save_dict, f)
        
  shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str, required=True, help='Path to the source directory containing input images')
  parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for saving LightGlue model outputs')
  parser.add_argument('--save_score', type=str, required=False, default=False, help='Whether to save score of matching between pixels')
  args = parser.parse_args()
  print(args)
  run(args.source_dir, args.output_dir, bool(args.save_score))
