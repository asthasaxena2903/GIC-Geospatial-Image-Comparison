import re
import os
import pandas as pd
import numpy as np
import math
import utm
import subprocess
from sklearn.neighbors import KDTree
from assets.utils import load_image, rbd
import torch
from typing import List, Tuple, Dict, Union

def radian_to_magnitude(lat_lon):
  return lat_lon[-1]*(lat_lon[0]+lat_lon[1]/60+lat_lon[2]/3600)

def fetch_meta_data(directory: str) -> Tuple[np.ndarray, List[str]]:
  """
    Extract metadata from image files in a given directory using ExifTool.

    Args:
        directory (str): The path to the directory containing image files.

    Returns:
        tuple:
            - numpy.ndarray: A 2D NumPy array containing the metadata for each image file.
              Each row represents an image file and contains the following information:
                  - x_utm (float): The x-coordinate in the UTM coordinate system.
                  - y_utm (float): The y-coordinate in the UTM coordinate system.
                  - extent (float): The estimated extent or coverage area of the image.
            - list: A list of file names (including extensions) in the directory.

    Raises:
        subprocess.CalledProcessError: If the ExifTool command fails to execute.

    Notes:
        - This function requires ExifTool to be installed and accessible from the command line.
        - The metadata extracted includes GPS position, relative altitude, and field of view.
        - The GPS position is converted to UTM coordinates using the `utm` library.
        - The extent is calculated based on the relative altitude and field of view.
  """

  files = sorted(os.listdir(directory))
  data = []
  for i in files:
    path = os.path.join(directory, i)
    metadata = str(subprocess.run(f"exiftool {path}", shell = True,  check = True, capture_output=True).stdout).split('\\n')
    metadata[0] = metadata[0].replace('b\'','')
    metadata = metadata[:-1]
    metadata = parse_exif_data(metadata)
    position = metadata['GPS Position']
    lat, lon = get_lat_lon_str(position)
    lat_, lon_ = radian_to_magnitude(lat), radian_to_magnitude(lon)
    x_utm, y_utm = utm.from_latlon(lat_, lon_)[:2]
    print("Metadata ======= ",metadata["Relative Altitude"][1:].replace("\\r", ""))
    extent = 2 * float(metadata["Relative Altitude"][1:].replace("\\r", "")) * math.tan(math.radians(float(metadata['Field Of View'][:3].strip())) / 2)
    data.append([x_utm, y_utm, extent])
  
  return np.array(data), files


def parse_exif_data(exif_output: List[str]) -> Dict[str, Union[str, int, float]]:
  """
    Parse the output of ExifTool and extract key-value pairs of EXIF data.

    Args:
        exif_output (List[str]): A list of strings representing the output of ExifTool.

    Returns:
        Dict[str, Union[str, int, float]]: A dictionary containing the parsed EXIF data,
            where the keys are EXIF tags, and the values are either strings, integers, or floats.

    Notes:
        - The function assumes that the ExifTool output is in the format "Key: Value".
        - It attempts to convert the value to an integer or float if the value matches certain patterns.
        - If the value cannot be converted to an integer or float, it remains a string.
        - The function removes the trailing period (.) from the values.
  """

  exif_data = {}
  for line in exif_output:
    key, value = re.split(r"\s*:\s*", line, maxsplit=1)
    value = value.rstrip('.')

    try:
      if re.search(r"^\d+\.\d+|\d+ (?:mm|cm|m|in|ft)", value):
        value = float(value.replace(" ", ""))
      elif re.search(r"^\d+$|1\/\d+$", value):
        value = int(value.replace("/", ""))
    except ValueError:
      pass

    exif_data[key] = value

  return exif_data

def get_lat_lon_str(string: str) -> Tuple:
  """
    Parse a string representation of GPS coordinates into latitude and longitude components.

    Args:
        string (str): A string representing GPS coordinates in the format "latitude, longitude".
            The latitude and longitude components should be in the format "degrees deg minutes' seconds" direction".
            For example: "48 deg 21' 43.58" N" or "2 deg 20' 53.95" E".

    Returns:
        Tuple[Tuple[float, float, float, int], Tuple[float, float, float, int]]:
            A tuple containing two tuples representing the latitude and longitude components, respectively.
            Each tuple contains:
                - degrees (float): The degree component of the coordinate.
                - minutes (float): The minute component of the coordinate.
                - seconds (float): The second component of the coordinate.
                - direction (int): The direction indicator (1 for North/East, -1 for South/West).

    Raises:
        ValueError: If the input string is not in the expected format.

    Notes:
        - The function assumes that the input string is in the correct format.
        - The direction indicator is represented as 1 for North/East and -1 for South/West.
  """

  lat, lon = string.split(', ')
  def lat_lon_helper(string_):
    deg, part2 = string_.split(' deg ')
    part2 = part2.split('\'')
    minute, part2 = float(part2[0][:2]), part2[1].split("\"")
    second, direction = float(part2[0].lstrip()), part2[1].lstrip()

    if direction in ['N', 'E']:
      direction = 1
    else:
      direction = -1

    return float(deg), minute, second, direction

  return lat_lon_helper(lat), lat_lon_helper(lon)

def fetch_neighbours(data: np.ndarray, radius: float) -> List[List[int]]:
  """
    Find the nearest neighbors for each point in the given data within a specified radius.

    Args:
        data (numpy.ndarray): A 2D NumPy array containing the data points, where each row represents a point in the format [x, y].
        radius (float): The radius within which to search for neighbors.

    Returns:
        list: A list of lists, where each inner list contains the indices of the nearest neighbors for the corresponding data point.

    Notes:
        - The function uses the `scipy.spatial.KDTree` to efficiently find the nearest neighbors.
        - A point is not considered a neighbor of itself.
  """

  tree = KDTree(data)
  nearest_neighbours = tree.query_radius(data, radius)
  nearest_neighbours = [[i for i in nearest_neighbours[j] if i!=j] for j in range(len(nearest_neighbours))]

  return nearest_neighbours

def generate_pairs(neighbours: List[List[int]]) -> List[Tuple[int, int]]:
  """
    Generate pairs of indices from the given list of neighbors.

    Args:
        neighbours (list): A list of lists, where each inner list contains the indices of the neighbors for a data point.

    Returns:
        list: A list of tuples, where each tuple represents a pair of indices (i, j) such that j is a neighbor of i.

    Notes:
        - The function ensures that each pair is unique and not duplicated (e.g., (i, j) and (j, i) are considered the same).
  """

  pairs = []

  for i in range(len(neighbours)):
    for j in neighbours[i]:
      if (j,i) not in pairs:
        pairs.append((i,j))

  return pairs

def get_neighbours(image_folder: str) -> Tuple[List[Tuple[int, int]], List[str]]:
  """
    Find the neighboring pairs of images in the given folder based on their metadata.

    Args:
        image_folder (str): The path to the folder containing the image files.

    Returns:
        tuple:
            - list: A list of tuples, where each tuple represents a pair of indices (i, j) of neighboring images.
            - list: A list of file names (including extensions) in the image folder.

    Notes:
        - The function uses the `fetch_meta_data` function to retrieve the metadata for the images.
        - The maximum extent among the images is used as the search radius for finding neighbors.
        - The `fetch_neighbours` and `generate_pairs` functions are used to find and generate the neighboring pairs, respectively.
  """

  meta_data, files = fetch_meta_data(image_folder)
  radius = np.max(meta_data[:,2])
  nearest_neighbours = fetch_neighbours(meta_data[:,:2], radius)
  nearest_neighbours = generate_pairs(nearest_neighbours)

  return nearest_neighbours, files

class ImageDataset():
  """
    A PyTorch dataset class for loading and preprocessing images for the SuperPoint model.

    This class is designed to load image files from a specified directory, preprocess them,
    and return the preprocessed images as tensors. It can be used in conjunction with PyTorch's
    DataLoader to efficiently load and batch data for training or inference.

    Attributes:
        root_dir (str): The path to the directory containing the image files.
        image_paths (list): A list of paths to the image files in the root directory.

    Args:
        root_dir (str): The path to the directory containing the image files.
  """

  def __init__(self, root_dir):
      self.root_dir = root_dir
      self.image_paths = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith('.JPG')]

  def __len__(self):
      return len(self.image_paths)

  def __getitem__(self, idx):
      image_path = self.image_paths[idx]
      image = load_image(image_path)
      return image

def save_output_temp(output_dict: dict, output_dir: str, filename: str) -> None:
  """
    Save the output dictionary to a file in the specified directory.

    Args:
        output_dict (dict): The dictionary containing the output data to be saved.
        output_dir (str): The path to the directory where the output file will be saved.
        filename (str): The filename for the output file (without extension).

    Notes:
        - The output file will be saved in PyTorch's '.pt' format.
        - The output dictionary is first moved to the CPU before saving to ensure compatibility.
  """

  output_path = os.path.join(output_dir, filename)+'.pt'
  output_dict = change_device(output_dict, 'cpu')
  torch.save(output_dict, output_path)

def save_output_macthes(output_dict: dict, output_dir: str, filename: str) -> None:
  """
    Save the output dictionary to a file in the specified directory.

    Args:
        output_dict (dict): The dictionary containing the output data to be saved.
        output_dir (str): The path to the directory where the output file will be saved.
        filename (str): The filename for the output file (without extension).

    Notes:
        - The output file will be saved in PyTorch's '.pt' format.
  """

  output_path = os.path.join(output_dir, filename)+'.pt'
  torch.save(output_dict, output_path)

def change_device(dict_: dict, device: Union[str, torch.device]) -> dict:
  """
    Move the PyTorch tensors in a dictionary to the specified device.

    Args:
        dict_ (dict): The dictionary containing PyTorch tensors.
        device (Union[str, torch.device]): The target device to move the tensors to.

    Returns:
        dict: The updated dictionary with tensors moved to the specified device.
  """

  for key in dict_.keys():
    if isinstance(dict_[key], torch.Tensor):
      dict_[key] = dict_[key].to(device)
  return dict_

def read_feature(temp_dir: str, filename: str, device: Union[str, torch.device]) -> dict:
  """
    Read a feature dictionary from a file and move it to the specified device.

    Args:
        temp_dir (str): The directory path where the feature file is located.
        filename (str): The filename for the feature file (without extension).
        device (Union[str, torch.device]): The target device to move the tensors to.

    Returns:
        dict: The feature dictionary with tensors moved to the specified device.
  """

  feature_path = os.path.join(temp_dir, filename) + ".pt"
  feature = torch.load(feature_path)
  feature = change_device(feature, device)

  return feature

def remove_dim(feature_list: List[torch.Tensor]) -> List:
  """
    Remove the batch dimension from a list of PyTorch tensors.

    Args:
        feature_list (List[torch.Tensor]): A list of PyTorch tensors with batch dimensions.

    Returns:
        List[Any]: A list of tensors or other objects without batch dimensions.

    Notes:
        - The function assumes that the input tensors have a batch dimension.
        - The `rbd` function is a separate function that removes the batch dimension from a tensor.
        - The function returns a list of the same length as the input, with the batch dimension removed.
  """
  
  return [rbd(x) for x in feature_list]

def get_matches(output_list: List, save_score: bool) -> List:
  """
    Extract matches between two images from the given output_list.

    Args:
        output_list (list): A list of dictionaries containing information about keypoints, matches, and scores.
            The expected structure is:
            [
                {'keypoints': ...},  # Keypoints for the first image
                {'keypoints': ...},  # Keypoints for the second image
                {'matches': ..., 'scores': ...}  # Matches and scores between the two images
            ]
        save_score (bool): If True, the function will include the match scores in the output.

    Returns:
        list: A list of tuples representing the matches between the two images.
            If save_score is True, each tuple contains (point1, point2, score).
            If save_score is False, each tuple contains (point1, point2).
  """
  
  matches_1 = output_list[2]['matches'][..., 0]
  matches_2 = output_list[2]['matches'][..., 1]

  image1_points = output_list[0]['keypoints'][matches_1].tolist()
  image2_points = output_list[1]['keypoints'][matches_2].tolist()

  matches_lis = []

  if save_score:
    scores = output_list[2]['scores'].tolist()
    compressed_matches = zip(image1_points, image2_points, scores)
  else:
    compressed_matches = zip(image1_points, image2_points)

  for match_ in compressed_matches:
      matches_lis.append(match_)

  return matches_lis












  