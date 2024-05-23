# The Point Net Suite

## Download and process datasets
* Dataset [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip):
```bash
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip --no-check-certificate
unzip point_net_suite/data/modelnet40_normal_resampled
rm point_net_suite/data/modelnet40_normal_resampled.zip
```

* Dataset [S3DIS](http://buildingparser.stanford.edu/dataset.html): 3D indoor parsing dataset
**TODO**: Update this with the URL that was used and with the script to process the data.

## Installation
You can work with anaconda or create a python3 virtual environment.
```bash
git clone https://github.com/jeferal/point_net_suite.git
cd point_net_suite
pip3 install -r requirements.txt
pip3 install -e .
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

