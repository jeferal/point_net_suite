''' Dataset for The aligned, reduced, partitioned S3DIS dataset 
    Provides functionality for train/test on partitioned sets as well 
    as testing on entire spaces via get_random_partitioned_space()
'''

import os
import random
import numpy as np
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import Dataset


class S3DIS(Dataset):
    CATEGORIES = [
        'ceiling',
        'floor',
        'wall',
        'beam',
        'column',
        'window',
        'door',
        'table',
        'chair',
        'sofa',
        'bookcase',
        'board',
        'stairs',
        'clutter',
    ]

    def __init__(self, root, area_nums, split='train', npoints=4096, r_prob=0.25, include_rgb=False, weight_type=None):
        self.root = root
        self.area_nums = area_nums      # i.e. '1-4' # areas 1-4
        self.split = split.lower()      # use 'test' in order to bypass augmentations
        self.npoints = npoints          # use  None to sample all the points
        self.r_prob = r_prob            # probability of rotation
        self.include_rgb = include_rgb  # include rgb info when returning the points (xyz + rgb)

        areas = []
        for area_num in area_nums:
            # glob all paths
            areas.append(os.path.join(root, f'Area_{area_num}'))

        # check that datapaths are valid, if not raise error
        for area_path in areas:
            if not os.path.exists(area_path):
                raise FileNotFoundError(f"PATH NOT VALID: {area_path} \n")

        # get all datapaths
        # data paths should be a list of all the txt files in the area folders
        self.data_paths = []
        for area_path in areas:
            if os.path.exists(area_path) and os.path.isdir(area_path):
                for room, _, files in os.walk(area_path):
                    if room == area_path:
                        continue  # Skip files directly in the area directory
                    for file in files:
                        self.data_paths.append(os.path.join(room, file))

        # get unique space identifiers (area_##\\spacename_##_)
        # TODO: This does not work
        self.space_ids = []
        for fp in self.data_paths:
            area = os.path.basename(fp)
            space = os.path.basename(fp)
            space_id = '\\'.join([area, '_'.join(space.split('_')[:2])]) + '_'
            self.space_ids.append(space_id)

        self.space_ids = list(set(self.space_ids))

        if weight_type == 'Sklearn':
            if self.split != 'test':
                labels_path = os.path.join(root, 'label_weights_sk.txt')
                if os.path.exists(labels_path):
                    self.labelweights = np.loadtxt(labels_path)
                else:
                    all_labels = np.array([], dtype=int)
                    for room_path in self.data_paths:
                        room_data = np.loadtxt(room_path)  # xyzrgbl
                        all_labels = np.append(all_labels, room_data[:, 6].astype(int))
                    self.labelweights = np.float32(compute_class_weight(class_weight="balanced", classes=np.unique(all_labels), y=all_labels))
                    #print(self.labelweights)
                    np.savetxt(labels_path, self.labelweights)
            else:
                self.labelweights = None

        elif weight_type == 'Custom':
            if self.split != 'test':
                labels_path = os.path.join(root, 'label_weights_custom.txt')
                if os.path.exists(labels_path):
                    self.labelweights = np.loadtxt(labels_path)
                else:
                    cat_weights = np.zeros(len(self.CATEGORIES))
                    for room_path in self.data_paths:
                        room_data = np.loadtxt(room_path)  # xyzrgbl
                        labels = room_data[:, 6]
                        tmp, _ = np.histogram(labels, range(len(self.CATEGORIES) + 1))
                        cat_weights += tmp
                    cat_weights = cat_weights.astype(np.float32)
                    cat_weights = cat_weights / np.sum(cat_weights)
                    self.labelweights = np.power(np.amax(cat_weights) / cat_weights, 1 / 3.0)
                    #print(self.labelweights)
                    np.savetxt(labels_path, self.labelweights)
            else:
                self.labelweights = None

        else:
            self.labelweights = None

    def __getitem__(self, idx):
        space_data = np.loadtxt(self.data_paths[idx])
        if self.include_rgb:
            points = space_data[:, :6]      # xyz points + rgb info
        else:
            points = space_data[:, :3]      # xyz points
        targets = space_data[:, 6]      # integer categories aster the rgb values

        # down sample point cloud
        if self.npoints:
            points, targets = self.downsample(points, targets)
            #points, targets = self.downsample_farthest_point(points, targets)

        # add Gaussian noise to point set if not testing
        if self.split != 'test':
            # add N(0, 1/100) noise
            points += np.random.normal(0., 0.01, points.shape)

            # add random rotation to the point cloud with probability
            if np.random.uniform(0, 1) > 1 - self.r_prob:
                points[:, :3] = self.random_rotate(points[:, :3])


        # Normalize Point Cloud to (0, 1)
        points = self.normalize_points(points)

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, targets

    def get_random_partitioned_space(self):
        ''' Obtains a Random space. In this case the batchsize would be
            the number of partitons that the space was separated into.
            This is a special function for testing.
        '''

        # get random space id
        idx = random.randint(0, len(self.space_ids) - 1)
        space_id = self.space_ids[idx]

        # get all filepaths for randomly selected space
        space_paths = []
        for fpath in self.data_paths:
            if space_id in fpath:
                space_paths.append(fpath)
        
        # assume npoints is very large if not passed
        if not self.npoints:
            self.npoints = 20000

        points = np.zeros((len(space_paths), self.npoints, 3))
        targets = np.zeros((len(space_paths), self.npoints))

        # obtain data
        for i, space_path in enumerate(space_paths):
            space_data = np.loadtxt(space_path)
            _points = space_data[:, :3] # xyz points
            _targets = space_data[:, 3] # integer categories

            # downsample point cloud
            _points, _targets = self.downsample(_points, _targets)

            # add points and targets to batch arrays
            points[i] = _points
            targets[i] = _targets

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, targets

    def downsample(self, points, targets):
        if len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are less points than the desired number
            choice = np.random.choice(len(points), self.npoints, replace=True)

        points = points[choice, :] 
        targets = targets[choice]

        return points, targets
    
    def downsample_farthest_point(self, points, targets):
        N, D = points.shape
        xyz = points[:,:3]
        centroids = np.zeros((self.npoints,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(self.npoints):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        choice = centroids.astype(np.int32)

        points = points[choice, :] 
        targets = targets[choice]

        return points, targets

    @staticmethod
    def random_rotate(points):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
        '''
        # construct a randomly parameterized 3x3 rotation matrix
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)

        rot_x = np.array([
            [1,              0,                 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi) ]])

        rot_y = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,                 1,                0],
            [-np.sin(theta), 0, np.cos(theta)]])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi),  0],
            [0,              0,                 1]])

        # rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))
        
        return np.matmul(points, rot_z)

    @staticmethod
    def normalize_points(points):
        ''' Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
            '''
        points = points - points.min(axis=0)
        points /= points.max(axis=0)

        return points

    def __len__(self):
        return len(self.data_paths)

    def get_categories(self):
        return self.CATEGORIES
