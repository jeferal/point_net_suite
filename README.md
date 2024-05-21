# The Point Net Suite

## Download and process datasets
* [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
```bash
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip --no-check-certificate
unzip point_net_suite/data/modelnet40_normal_resampled
rm point_net_suite/data/modelnet40_normal_resampled.zip
```

* [S3DIS](http://buildingparser.stanford.edu/dataset.html): 3D indoor parsing dataset

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```
