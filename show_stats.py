import torch
import argparse
from pathlib import Path


models_log_folder_dict = {'pointnet_cls': 'classification/pointnet_cls',
                          'pointnet2_cls_ssg': 'classification/pointnet_v2_cls_ssg',
                          'pointnet2_cls_msg': 'classification/pointnet_v2_cls_msg',
                          'pointnet_sem_seg': 'semantic_seg/pointnet_sem_segmentation',
                          'pointnet2_sem_seg_ssg': 'semantic_seg/pointnet_v2_sem_seg_ssg',
                          'pointnet2_sem_seg_msg': 'semantic_seg/pointnet_v2_sem_seg_msg',
                          'random': 'semantic_seg'}

def parse_args():
    parser = argparse.ArgumentParser('show_stats')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_ssg', help='model name [default: pointnet_cls]')

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
            print('train_loss =', end=" ")
            print(checkpoint['train_loss'])
            print('#LEARNING RATE:')
            print('learning_rate =', end=" ")
            print(checkpoint['optim_learning_rate'])
            print('#TRAIN ACCURACY:')
            print('train_acc =', end=" ")
            print(checkpoint['train_accuracy'])
            try:
                print('#TRAIN IOU:')
                train_iou = checkpoint['train_iou']
                print('train_iou =', end=" ")
                print(train_iou)
            except:
                print('#This checkpoints does not have train_iou.')
            print('#EVAL LOSS:')
            print('eval_loss =', end=" ")
            print(checkpoint['eval_loss'])
            print('#EVAL ACCURACY:')
            print('eval_acc =', end=" ")
            print(checkpoint['eval_accuracy'])
            try:
                print('#EVAL MEAN CLASS ACCURACY:')
                eval_class_acc = checkpoint['test_mean_class_accuracy']
                print('eval_class_acc =', end=" ")
                print(eval_class_acc)
            except:
                print('#This checkpoints does not have test_mean_class_accuracy.')
            try:
                print('#EVAL IOU:')
                eval_iou = checkpoint['eval_iou']
                print('eval_iou =', end=" ")
                print(eval_iou)
            except:
                print('#This checkpoints does not have eval_iou.')
        except:
            print('Checkpoint does not exist...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
