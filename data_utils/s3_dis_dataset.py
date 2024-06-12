''' Dataset for The aligned, reduced, partitioned S3DIS dataset 
    Provides functionality for train/test on partitioned sets as well 
    as testing on entire spaces via get_random_partitioned_space()
    '''

import os
from glob import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class S3DIS(Dataset):
    def __init__(self, root, area_nums, split='train', npoints=4096, r_prob=0.25, include_rgb=False):
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
                for root, _, files in os.walk(area_path):
                    if root == area_path:
                        continue  # Skip files directly in the area directory
                    for file in files:
                        self.data_paths.append(os.path.join(root, file))

        # get unique space identifiers (area_##\\spacename_##_)
        # TODO: This does not work
        self.space_ids = []
        for fp in self.data_paths:
            area = os.path.basename(fp)
            space = os.path.basename(fp)
            space_id = '\\'.join([area, '_'.join(space.split('_')[:2])]) + '_'
            self.space_ids.append(space_id)

        self.space_ids = list(set(self.space_ids))

        '''if self.split != 'test'::
            labelweights = np.zeros(13)
            for room_path in self.data_paths:
                room_data = np.loadtxt(room_path)  # xyzrgbl
                labels = room_data[:, 6]
                tmp, _ = np.histogram(labels, range(14))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
            #print(self.labelweights)
        else:
            self.labelweights = None'''
        if self.split != 'test':
            label_probs = np.zeros(14)
            for room_path in self.data_paths:
                room_data = np.loadtxt(room_path)  # xyzrgbl
                targets = room_data[:, 6]
                unique_targets, counts_targets = np.unique(targets, return_counts=True)
                for i, target in enumerate(unique_targets):
                    label_probs[int(target)] += counts_targets[i]
            label_probs = label_probs.astype(np.float32)
            label_probs = label_probs / np.sum(label_probs)
            beta_value = 3.0
            self.label_probs_softmax = np.exp(beta_value * label_probs) / np.sum(np.exp(beta_value * label_probs))
            #print(label_probs)
            #print(self.label_probs_softmax)
        else:
            self.label_probs_softmax = np.ones(14) / 14

    def __getitem__(self, idx):
        space_data = np.loadtxt(self.data_paths[idx])
        if self.include_rgb:
            points = space_data[:, :6]      # xyz points + rgb info
        else:
            points = space_data[:, :3]      # xyz points
        targets = space_data[:, 6]      # integer categories aster the rgb values

        # down sample point cloud
        if self.npoints:
            if self.split != 'test':
                points, targets = self.downsample_with_label_probs_softmax(points, targets)
            else:
                points, targets = self.downsample(points, targets)

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
    
    def downsample_with_label_probs_softmax(self, points, targets):
        unique_targets = np.unique(targets)
        probs_for_target = self.label_probs_softmax.copy()

        # Share probabilities between the rest of targets if a target does not exist in this instance
        empty_targets = []
        for i in range(len(probs_for_target)):
            if i not in unique_targets:
                empty_targets.append(i)

        probability_to_split = 0
        for empty_target in empty_targets:
            probability_to_split += probs_for_target[empty_target]
            probs_for_target[empty_target] = 0.0

        prob_to_add_each_target = probability_to_split / len(unique_targets)
        for target in unique_targets:
            probs_for_target[int(target)] += prob_to_add_each_target

        # Calculate the number of points to sample for each target
        points_to_sample_per_target = np.zeros(14)
        total_allocated_points = 0
        
        for target in unique_targets:
            # Calculate the number of points to sample for this target
            num_points_to_sample = int(np.floor(self.npoints * probs_for_target[int(target)]))
            points_to_sample_per_target[int(target)] = num_points_to_sample
            total_allocated_points += num_points_to_sample
        
        # Add the remaining number of points to ensure the total is exactly self.npoints
        while total_allocated_points < self.npoints:
            for target in unique_targets:
                if total_allocated_points < self.npoints:
                    points_to_sample_per_target[int(target)] += 1
                    total_allocated_points += 1
                else:
                    break

        # Initialize lists to collect sampled points and targets
        choice = np.array([], dtype=int)
        #print("Starting sample:")
        for target in unique_targets:
            # Get the indices of all points belonging to the current target
            target_indices = np.where(targets == target)[0]

            num_points_to_sample = int(points_to_sample_per_target[int(target)])

            #print(str(len(target_indices)) + " - " + str(num_points_to_sample))
            
            if len(target_indices) >= num_points_to_sample:
                target_choice = np.random.choice(target_indices, num_points_to_sample, replace=False)
            else:
                target_choice = np.random.choice(target_indices, num_points_to_sample, replace=True)
            
            choice = np.concatenate((choice, target_choice), axis=None)
        
        np.random.shuffle(choice)

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
