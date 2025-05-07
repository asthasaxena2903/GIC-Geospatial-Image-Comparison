**Image Matching Pipeline with LightGlue and SuperPoint**

This repository provides a Python pipeline for efficiently matching images using the SuperPoint keypoint extractor and LightGlue matcher. It leverages EXIF information, specifically GPS data, to identify image pairs for comparison.

**Features:**

- **Selective Matching:** Utilizes EXIF GPS data to determine which image pairs require matching, optimizing processing time.
- **SuperPoint & LightGlue:** Employs the state-of-the-art SuperPoint extractor for fast and accurate keypoint detection and LightGlue for efficient feature matching.
- **Flexible Output:** Stores the results as a JSON file, including matching scores if specified.
- **Command-Line Usage:** Easy execution via a Python script with clear command-line arguments.

**Installation:**

1. **Clone the Repository:**
   ```bash
   clone the directory
   ```

2. **Install Dependencies:**
   Ensure you have Python and pip installed. Then, install the required packages:
   ```bash
   cd Lightglue_pipeline  
   pip install -r requirements.txt
   ```

3. **Install EXIF Library (if not already installed):**
   On Ubuntu/Debian-based systems:
   ```bash
   sudo apt install libimage-exiftool-perl
   ```

**Running the Pipeline:**

Use the `run.py` script with the following command-line arguments:

- `--source_dir`: Path to the directory containing the input images.
- `--output_dir`: Path to the directory where the JSON output will be saved.
- `--save_score` (optional): Boolean value (True or False) to determine if match scores are included in the output (default: False).

**Example:**

```bash
python run.py --source_dir /path/to/images --output_dir /path/to/output --save_score True
```

This command will match images in the `/path/to/images` directory, save the results as a JSON file in the `/path/to/output` directory, and include match scores in the output.

**Explanation:**

The `run.py` script performs the following steps:

1. Loads image pairs and filenames using the `get_neighbours` function.
2. Sets up the device (CPU or GPU) based on available resources.
3. Initializes SuperPoint and LightGlue models.
4. Creates a temporary directory for storing intermediate feature files.
5. Iterates through each image:
   - Extracts keypoints and features using SuperPoint.
   - Saves features to a temporary file.
6. Processes each image pair:
   - Loads features from temporary files.
   - Matches features using LightGlue.
   - Extracts matches and formats the output.
   - Saves matches in a dictionary with the image pair as the key.
7. Saves the matches dictionary as a JSON file in the output directory.
8. Removes the temporary directory.
# Image Similarity Matching

## Overview
This project performs image similarity matching using deep learning-based feature extraction. Given an index image and a set of response images, it computes similarity scores and visualizes the results.

## Project Structure
```
project_root/
│── dataset/                # Store index & response images
│   ├── index_image.JPG     # Index image
│   ├── response/           # Folder containing response images
│── output/                 # Store generated results & visualizations
│── utils.py            # Processing, feature extraction & visualization functions
│── run.py          # Main script for execution
│── requirements.txt        # Dependencies
│── readme.md               # Documentation
```

## Installation
1. Clone the repository:
   ```bash
   clone the directory
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script with the following command:
```bash
python run.py --index_image dataset/index_image.JPG --response_path dataset/response --output_path output
```

## Dependencies
List of required packages (available in `requirements.txt`):
- `numpy`
- `pandas`
- `torch`
- `torchvision`
- `tqdm`
- `Pillow`
- `matplotlib`
- `seaborn`

## Features
- Extracts features from images using a pre-trained ResNet model
- Computes cosine similarity between images
- Saves results in JSON and CSV format
- Generates visual comparisons and histogram plots

## Output
- `output/output.json`: JSON file containing similarity scores
- `output/sorted_results.csv`: CSV file with sorted similarity scores
- `output/comparisons/`: Directory with comparison images
- `output/score_histogram.png`: Histogram of similarity scores

## License
This project is open-source and available under the MIT License.

