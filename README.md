# AIDL24: Towards improving minority classes accuracy for Aerial Lidar Datasets
TBC With Mariona!!!

**Abstract here**

### About
Final project of the Postgraduate Course in Artificial Intelligence with Deep Learning. Authors:
- Sergio Calzón Molina
- Flavia Diletta Caudana
- Jesús Ferrándiz Alarcón
- Jorge Ruiz Vázquez

Supervisor:
- Mariona Carós Roca

## Table of Contents <a name="toc"></a>

- [1. Introduction](#1-intro)
    - [1.1. Motivation](#11-motivation)
    - [1.2. Milestones](#12-milestones)

- [2. Implementation](#2-Implementation)
    - [2.1. Data Preprocessing and Datasets](#21-data-preprocessing)
        - [2.1.3. Dales Dataset](#213-dales-dataset)
    - [2.2. Models](#22-models)
        - [2.2.3. PointNet](#223-pointnet)
        - [2.2.4. PointNet++](#224-pointnetpp)
    - [2.3. Sampling](#223-sampling)
    - [2.4. Experiments](#24-experiments)

- [3. Final application](#3-Final-application)
- [4. How to run the code](#4-How-to-run-the-code)
- [5. Conclusion](#5-conclusions)

- [6. Acknowledgements](#6-Acknowledgements)

## 1. Introduction <a name="1-intro"></a>

### 1.1. Motivation <a name="11-motivation"></a>

### 1.2. Milestones <a name="12-milestones"></a>

## 2. Implementation <a name="2-Implementation"></a>

### 2.1. Data Preprocessing and Datasets <a name="21-data-preprocessing"></a>

### 2.1.1. ModelNet <a name="211-dales-dataset"></a>
The ModelNet dataset is a widely-used dataset for 3D object classification and recognition. It is similar to the MNIST of pointcloud object classification. It includes a comprehensive collection of 3D CAD models, featuring 662 object categories ranging from everyday items like chairs, guns and tables to complex structures like airplanes and cars. The dataset is divided into two main subsets: ModelNet10 and ModelNet40. ModelNet10 includes 10 categories with around 5,000 models, while ModelNet40 comprises 40 categories with approximately 12,000 models. Each model is consistently scaled and oriented, ensuring a standardized basis for algorithm comparison which is paramount for speeding up the learning process in pointcloud classification.

**Key Features:**

- Categories: 662 object categories.
- Subsets: ModelNet10 (10 categories) and ModelNet40 (40 categories).
- Number of Models: Over 17,000 CAD models.
- Applications: 3D object classification, recognition, and retrieval.

**What have we learned?**

After working with the ModelNet dataset, we've learned that not all points in a point cloud are equally important. Some points carry more significant information and are crucial for classification tasks. These critical points can hugely impact the performance of Deep Learning models. Identifying and focusing on these key points is essential, especially when dealing with large point clouds as we faced on the DALES dataset.

The importance of critical points becomes even more apparent in the context of downsampling techniques. Downsampling helps on reducing the computational load by decreasing the number of points in a point cloud while retaining the most informative ones. This process is paramount when feeding very large point clouds to networks like PointNet++ for training, as it ensures that the network focuses on the most relevant features, leading to more efficient and accurate learning and relevant feature extraction.



### 2.1.2. Stanford indoors 3D <a name="212-dales-dataset"></a>
The Dales Objects dataset is a Large Scale Benchmark Dataset for Segmentation and 
Instance Segmentation of Aerial Lidar data. It contains close to half-bilion hand labeled points and the dataset covers over 10 square kilometers. Each point also contains an intensity value. The dataset contains the following classes with the following number of points:

### 2.1.3. Dales Dataset <a name="213-dales-dataset"></a>
The Dales Objects dataset is a Large Scale Benchmark Dataset for Segmentation and 
Instance Segmentation of Aerial Lidar data. It contains close to half-bilion hand labeled points and the dataset covers over 10 square kilometers. Each point also contains an intensity value. The dataset contains the following classes with the following number of points:

| Class Name  | Number of Points |
|-------------|------------------|
| Ground      | 246.9M           |
| Vegetation  | 159M             |
| Car         | 4.1M             |
| Truck       | 879k             |
| Powerline   | 994k             |
| Fence       | 2.1M             |
| Pole        | 262k             |
| Buildings   | 78.7M            |

If we attend to the number of points per class, we can see that the dataset is highly unbalanced. Most of the points belong to the 
classes ground and vegetation, while the classes truck, powerline, fence and pole have very few points in comparison, effectively making them part of a group called minority classes. Minority classes are more difficult to train and normally present lower convergence ratios. This is a problem
that we will try to address in this project.

The dataset is divided into 2 splits: train and test. The train split contains 58 .ply files, which consist of tiles of 500 x 500 meters and the test split contains 22 .ply files.

Each tile contains around 12M points. This is a very large number of points and it is not feasible to train a model with all
the points at once. First of all, we might not have enough GPU memory to store all the points, we would have to downsample the points
heavily and we would lose a lot of information. For these reasons we decided to partition the tiles into smaller tiles.

The partitiong of the tiles also depends in an overlap parameter [0-1], which determines how much overlap there is between the tiles
to ensure that the objects are not cut. Partitioning these large point clouds takes very long CPU time. We have devided a tile into
chunks so that each core of the CPU can process a chunk in parallel and store the information in a file. This way the process takes considerably less time.

The partitioning is part of the class Dataset, where if the dataset with a certain partition parameter and overlap is not found in 
disk, then it will create it and store it in disk. Every time the client calls the method getitem, then the dataset will load from disk the file of the particular partition for the given index and will return the points and the labels.

#### Images of the partitions

### 2.2. Models <a name="22-models"></a>

#### 2.2.3. PointNet <a name="223-pointnet"></a>

<p align="center">
  <img src="assets/pointnet.png">
  <br>
  <em>Figure <number>: PointNet architecture.</em>
</p>

#### 2.2.4. PointNet++ <a name="224-pointnetpp"></a>

<p align="center">
  <img src="assets/pointnetpp.png">
  <br>
  <em>Figure <number>: PointNet++ architecture.</em>
</p>

### 2.3. Sampling <a name="23-sampling"></a>

### 2.4. Experiments <a name="24-Experiments"></a>

#### 2.4.1. Experiment logging <a name="241-experiment-logging"></a>
We have conducted different experiments using different models and hyperparameters. We decided to log the metrics of every 
experiment in Mlflow because it is a widely used open source application that can help us keep track of the experiments, understand
the results and better choose the hyperparameters.

The Mlflow server was running in a Google Cloud instance so that every member of the team can access to it from any machine.
In order to run the mlflow server in the Google Cloud instance, the machine needs to have docker installed. After installing docker, 
the Mlflow application can be run like this:
```bash
docker run -it --name mlflow -p <host_port>:5000 -v <mlruns_host_path>:/mlruns -v <mlartifacts_host_path>:/mlartifacts ghcr.io/mlflow/mlflow:v2.13.0 mlflow ui --host 0.0.0.0
```
To access from any other machine via http, we have to change the firewall rules of the virtual machine in Google Cloud so that the port is exposed to the internet. Further work should involve changing the security of the communication to https.

For this experiment we have logged the following metrics:

**Classification task**:
- train_loss
- eval_loss
- train_mean_accuracy
- eval_mean_accuracy
- train_per_class_loss
- eval_per_class_loss

**Segmentation task**:
- train_loss
- eval_loss
- train_accuracy
- eval_accuracy
- train_per_class_iou
- eval_per_class_iou

We have also logged **system metrics** which some of them are:
- System memory usage
- GPU memory usage
- CPU utilization percentage

And other information that is helpful for us:
- A checkpoint of the best model so far in the format .pth
- A plot with train and eval classes distribution
- The command the script was run
- The arguments the train was created with for reproducibility 

#### 2.4.2. Experiments with Dales PointNet++ semantic segmentation <a name="242-experiments-dales-pointnetpp"></a>

**Experiment Base Hyper Parameters:**

| Learning Rate | Optimizer | Batch Size | Num Points | Grouping Method | Dropout | Scheduler | Label Smoothing | Extra Feature Dropout |
|---------------|-----------|------------|------------|-----------------|---------|-----------|------------------|-----------------------|
| 0.001         | AdamW     | 8          | 8192       | MSG             | 0.5     | Cosine    | 0.1              | 0.2                   |

**Hypothesis:**
- Describe the hypothesis here.

**Results:**
- Describe the results here.

**Conclusions:**
- Describe the conclusions here.



Intro talking about:
- checkpoints and info logged into .pth

Each experiment must contain:
1. Hypothesis
2. Experiment Setup / Implementation
3. Results
4. Conclusions

<p align="center">
  <img src="assets/mlflow.png">
  <br>
  <em>Figure <number>: Mlflow server.</em>
</p>


## 3. Final application <a name="3-Final-application"></a>
The Graphical User Interface

## 4. How to run the code <a name="4-How-to-run-the-code"></a>
It is mandatory to show how to run the code

## 5. Conclusions <a name="5-conclusions"></a>

## 6. Acknowledgements <a name="6-Acknowledgements"></a>



# The Point Net Suite

## Download and process datasets
* Dataset [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip):
```bash
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip --no-check-certificate
unzip modelnet40_normal_resampled.zip -d point_net_suite/data/modelnet40
rm modelnet40_normal_resampled.zip
```

* Dataset [S3DIS](http://buildingparser.stanford.edu/dataset.html): 3D indoor parsing dataset
```bash
wget https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip
unzip Stanford3dDataset_v1.2_Aligned_Version.zip -d data/stanford_indoor3d
rm Stanford3dDataset_v1.2_Aligned_Version.zip
```
Use the script to process the areas of the dataset:
```bash
python3 point_net_suite/data_utils/s3_dis_data_gen.py <path_to_the_Area>
```

## Installation

### Option 1: pip3 venv via requirements.txt 
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/jeferal/point_net_suite.git
    cd point_net_suite
    ```

2. **Install Requirements:**
    ```bash
    pip3 install -r requirements.txt
    pip3 install -e .
    ```
### Option 2: Conda Environment via yaml file

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/jeferal/point_net_suite.git
    cd point_net_suite
    ```

2. **Create the Conda Environment:**
    ```bash
    conda env create -f conda_env_backup.yaml
    ```

3. **Activate the Environment:**
    ```bash
    conda activate pointnet_thesis
    ```

## Train a model
```bash
python3 point_net_suite/train_classification.py
```

```bash
python3 point_net_suite/train_segmentation.py
```

The trainings will be monitored by mlflow if
using the argument --use_mlflow.

## Inference
Classification:
```bash
python3 scripts/inference_cls.py <model_path> <dataset_path>
```
![Alt text](./assets/point_net_inference_cls.png)

Semantic Segmentation:
![Alt text](./assets/point_net_inference_seg.png)
```bash
python3 scripts/inference_seg.py <model_path> <dataset_path>
```

## DALES dataset
The dales dataset is composed of files with very large point clouds. These
point clouds must be partitioned into tiles so that we can use them for training.
The script `scripts/visualize_dales.py` can be used to visualize the tiles of the
point clouds. A particular partition will be stored in disk remembering the parameters N (number of partitions per side) and overlap [0,1]. If the DALES dataset
is created with the same paramters it was used before, it will use the cache. You can
run the visualization with this command:
```bash
python3 scripts/visualize_dales.py data/DALESObjects <split_name> <index> --partitions <number_of_partitions> --overlap <overlap_from_0_to_1> --intensity
```
<img src="./assets/dales_tile_example_1.png" alt="Alt text" style="width:50%;">
<img src="./assets/dales_tile_example_2.png" alt="Alt text" style="width:50%;">

## Test
The tests are located in the test folder. All the tests can be run with the following command:
```bash
python -m unittest discover -s test -p 'test_*.py' -v
```
