# DLUP
A deep learning method for Human Plasma Proteome identification and quantification.

## Installation

### 1. Install Python (Anaconda)
Download and install Anaconda.

Create a new environment with Python 3.9+:

```bash
conda create --name yourEnv python=3.9
```

Activate the environment:

```bash
conda activate yourEnv
```

### 2. Install CUDA (optional)
If you have a GPU, you can install CUDA. If not, skip this step.

Visit the CUDA Toolkit Archive.

Select your computer's architecture.

Choose CUDA 11.7.0 and download the toolkit.

### 3. Install Required Packages

Install the necessary packages from the requirements.txt file:

```bash
conda install --yes --file requirements.txt
```

## Getting Started

### 1. Prepare Your Raw Data

Create a folder to store your raw data. Here is an example of how your folder structure should look:

```
ttp_20230702_CCS
├── ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-1_Slot1-10_1_12901.d
├── ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-2_Slot1-10_1_12902.d
├── ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-3_Slot1-10_1_12903.d
├── ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-4_Slot1-10_1_12904.d
└── ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-5_Slot1-10_1_12905.d
```

### 2. Generate the Library

Create a folder to store the raw library files with the .tsv suffix.

Run library_generation.py to generate the target and decoy libraries with the .npy suffix.

### 3. Create the Config Files
In the configs folder, you will find two JSON files:

-   [x] preprocess_configs.json
-   [x] model_configs.json

#### preprocess_configs.json

You can modify the following fields:

-   `library`: Replace target_path and decoy_path with your library paths.
-   `save_root_path`: Set the path to save the preprocessed data.
-   `mobilityDistanceThreshold`: Set to 0.05 or 0.1 to filter MS2 data.

#### model_configs.json

You can modify the batch_size field.

### 4. Identification and Quantification

Run the main.py script:

```bash
python main.py --data_path='raw_data_path' --num_process=5
```

The results, including the peptide and protein group quantification in TSV format, will be saved in the save_root_path folder.