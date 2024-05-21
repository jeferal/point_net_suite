import torch
import argparse
from pathlib import Path


models_log_folder_dict = {'pointnet_cls': 'classification/pointnet_cls',
                          'pointnet_sem_seg': 'semantic_seg/pointnet_sem_segmentation'}

def parse_args():
    parser = argparse.ArgumentParser('show_stats')
    parser.add_argument('--model', type=str, default='pointnet_cls', help='model name [default: pointnet_cls]')

    return parser.parse_args()

def main(args):
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(models_log_folder_dict[args.model])
    if exp_dir.exists():
        print('Folder exists!')
        try:
            checkpoint = torch.load(str(exp_dir) + '/best_model.pth')
            print('EPOCH:')
            print(checkpoint['epoch'])
            print('TRAIN LOSS:')
            print(checkpoint['train_loss'])
            print('TRAIN ACCURACY:')
            print(checkpoint['train_accuracy'])
            print('TRAIN IOU:')
            print(checkpoint['train_iou'])
            print('TEST LOSS:')
            print(checkpoint['test_loss'])
            print('TEST ACCURACY:')
            print(checkpoint['test_accuracy'])
            try:
                print('TEST MEAN CLASS ACCURACY:')
                print(checkpoint['test_mean_class_accuracy'])
            except:
                print('This checkpoints does not have test_mean_class_accuracy.')
            print('TEST IOU:')
            print(checkpoint['test_iou'])
        except:
            print('Checkpoint does not exist...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
