from data_utils.dales_dataset import DalesDataset

dales_dataset = DalesDataset(root='/home/jesusferrandiz/Learning/point_net_ws/src/point_net_suite/data/DALESObjects', partitions=5, split='train', intensity=False, instance_seg=False)
