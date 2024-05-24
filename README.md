# DLUP

a deep learning method for **Human Plasma** Proteome identification and quantification

## Installation

### 1. Install Python (Anaconda)

download and install [Anaconda](https://www.anaconda.com)

then create an environment

maybe you need the python 3.9+

```bash
conda create --n 'yourEnv' python=3.9
```

then activate the environment

```bash
conda activate 'yourEnv'
```

### 2. Install cuda

if you don't have a gpu, you can skip this step.

click the url [[CUDA Toolkit Archive | NVIDIA Developer]](https://developer.nvidia.com/cuda-toolkit-archive), then select the architecture of your computer, finally choose the cuda11.7.0, then download the cuda

### 3. Install packages

```bash
conda install --yes --file requirements.txt
```

## Getting Started

### 1. Prepare your raw data

you need to create a folder to store the raw data.

below is an example 

```
ttp_20230702_CCS
--- ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-1_Slot1-10_1_12901.d
--- ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-2_Slot1-10_1_12902.d
--- ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-3_Slot1-10_1_12903.d
--- ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-4_Slot1-10_1_12904.d
--- ttp_20230702_CCS_Plasma_trypsin_Mix1_20min_system-test-5_Slot1-10_1_12905.d
```

### 2. generate the library with '.npy' suffix

first, you should create a folder to store the raw library (should with the `.tsv` suffix)

Then run the library_generation.py to generate the target and the decoy library.

### 3. Create the config files

In the floder "configs", you can see the three json files.

```
-   preprocess_configs.json
-   model_configs.json
```

#### preprocess_configs.json

you can only modify the `library`, `save_root_path`, `mobilityDistanceThreshold`

-   replace the `target_path` and `decoy_path` with your library path
-   `save_root_path` is the path to save the preprocess data, you should replace that with you own path
-   `mobilityDistanceThreshold` is the mobilityDistanceThreshold (only filter with the ms2 that are satisfied), you can modify it 0.05 or 0.1.
  
#### model_configs.json

you can only modify the `batch_size`

### Identification and Quantification 

run the `main.py`

```bash
python main.py --data_path='raw_data path' num_process=5
```

then you can gain the tsv file of the peptide and protein group quantification in the `save_root_path` folder.