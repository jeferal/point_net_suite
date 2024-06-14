import torch
import argparse
from pathlib import Path


models_log_folder_dict = {'pointnet_cls': 'classification/pointnet_cls',
                          'pointnet_sem_seg': 'semantic_seg/pointnet_sem_segmentation',
                          'pointnet2_sem_seg_ssg': 'semantic_seg/pointnet_v2_sem_seg_ssg'}

def parse_args():
    parser = argparse.ArgumentParser('show_stats')
    parser.add_argument('--model', type=str, default='random', help='model name [default: pointnet_cls]')

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
            print('#TRAIN LOSS:')
            print('train_loss =')
            print(checkpoint['train_loss'])
            print('#TRAIN ACCURACY:')
            print('train_acc =')
            print(checkpoint['train_accuracy'])
            print('#TRAIN IOU:')
            print('train_iou =')
            print(checkpoint['train_iou'])
            print('#EVAL LOSS:')
            print('eval_loss =')
            print(checkpoint['eval_loss'])
            print('#EVAL ACCURACY:')
            print('eval_acc =')
            print(checkpoint['eval_accuracy'])
            try:
                print('#EVAL MEAN CLASS ACCURACY:')
                eval_class_acc = checkpoint['test_mean_class_accuracy']
                print('eval_class_acc =')
                print(eval_class_acc)
            except:
                print('#This checkpoints does not have test_mean_class_accuracy.')
            print('#EVAL IOU:')
            print('eval_iou =')
            print(checkpoint['eval_iou'])
        except:
            print('Checkpoint does not exist...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
