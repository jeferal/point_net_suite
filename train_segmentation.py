import os
import sys
import torch
import numpy as np
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader

#import data_utils.DataAugmentationAndShuffle as DataAugmentator
from point_net_suite.data_utils.metrics import compute_iou
from point_net_suite.data_utils.s3_dis_dataset import S3DIS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_folder_dict = {'pointnet_sem_segmentation': 'models/Pointnet'}
models_modules_dict = {'pointnet_sem_segmentation': 'point_net_suite.models.pointnet_sem_segmentation'}

CATEGORIES = {
    'ceiling'  : 0, 
    'floor'    : 1, 
    'wall'     : 2, 
    'beam'     : 3, 
    'column'   : 4, 
    'window'   : 5,
    'door'     : 6, 
    'table'    : 7, 
    'chair'    : 8, 
    'sofa'     : 9, 
    'bookcase' : 10, 
    'board'    : 11,
    'stairs'   : 12,
    'clutter'  : 13
}
NUM_CLASSES = len(CATEGORIES)


def parse_args():
    parser = argparse.ArgumentParser('training_sem_seg')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--model', type=str, default='pointnet_sem_segmentation', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    #parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD [default: Adam]')
    #parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--num_points', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--train_area', type=int, nargs='+', default=[1, 2, 4], help='Which area to use for test, option: 1-3 [default: [1,2,4]]')
    parser.add_argument('--test_area', type=int, nargs='+', default=[3], help='Which area to use for test, option: 1-4 [default: [3]]')
    parser.add_argument('--remove_checkpoint', action='store_true', default=False, help='remove last checkpoint train progress')

    return parser.parse_args()


def main(args):
    def log_string(str):
        #logger.info(str)
        print(str)

    #'''HYPER PARAMETER'''
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('semantic_seg')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.model)
    if args.remove_checkpoint and exp_dir.exists():
        log_string('Deleting existing checkpoint!')
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(exist_ok=True)
    
    log_string('PARAMETER ...')
    log_string(args)

    data_path = os.path.join(BASE_DIR, 'data/stanford_indoor3d')
    num_classes = NUM_CLASSES
    num_epochs = args.epoch
    num_points = args.num_points
    batch_size = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DIS(root=data_path, area_nums=args.train_area, npoints=num_points, r_prob=0.25)
    print("start loading test data ...")
    TEST_DATASET = S3DIS(root=data_path, area_nums=args.test_area, split='test', npoints=num_points)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    sys.path.append(os.path.join(BASE_DIR, models_folder_dict[args.model]))
    model = importlib.import_module(models_modules_dict[args.model])

    classifier = model.get_model(num_points=num_points, m=num_classes)
    criterion = model.get_loss()

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate, max_lr=1e-3, 
                                                    step_size_up=1000, cycle_momentum=False)
    
    best_iou = 0.2

    # Lists to store metrics
    train_loss = []
    train_accuracy = []
    train_iou = []
    test_loss = []
    test_accuracy = []
    test_iou = []

    try:
        checkpoint = torch.load(str(exp_dir) + '/best_model.pth')
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        train_iou = checkpoint['train_iou']
        test_loss = checkpoint['test_loss']
        test_accuracy = checkpoint['test_accuracy']
        test_iou = checkpoint['test_iou']
        best_iou = test_iou[-1]
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_type'] == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    global_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # Set the model in training mode
        classifier = classifier.train()
        _train_loss = []
        _train_accuracy = []
        _train_iou = []
        for i, (points, targets) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            
            if not args.use_cpu:
                points, targets = points.transpose(2, 1).cuda(), targets.squeeze().cuda()
            else:
                points, targets = points.transpose(2, 1), targets.squeeze()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # get predicted class logits
            preds, crit_idxs, feat_trans = classifier(points)

            # get class predictions
            pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

            # get loss and perform backprop
            loss = criterion(preds, targets, pred_choice) 
            loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate
            
            # get metrics
            correct = pred_choice.eq(targets.data).cpu().sum()
            accuracy = correct/float(batch_size*num_points)
            iou = compute_iou(targets, pred_choice)

            # update epoch loss and accuracy
            _train_loss.append(loss.item())
            _train_accuracy.append(accuracy)
            _train_iou.append(iou.item())
            
        train_loss.append(np.mean(_train_loss))
        train_accuracy.append(np.mean(_train_accuracy))
        train_iou.append(np.mean(_train_iou))

        print(f'Epoch: {epoch + 1} - Train Loss: {train_loss[-1]:.4f} ' \
            + f'- Train Accuracy: {train_accuracy[-1]:.4f} ' \
            + f'- Train IOU: {train_iou[-1]:.4f}')

        
        # get test results after each epoch
        with torch.no_grad():

            # Set model in evaluation mode
            classifier = classifier.eval()

            _test_loss = []
            _test_accuracy = []
            _test_iou = []
            for i, (points, targets) in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
                
                if not args.use_cpu:
                    points, targets = points.transpose(2, 1).cuda(), targets.squeeze().cuda()
                else:
                    points, targets = points.transpose(2, 1), targets.squeeze()

                preds, crit_idxs, feat_trans = classifier(points)
                pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

                loss = criterion(preds, targets, pred_choice) 

                # get metrics
                correct = pred_choice.eq(targets.data).cpu().sum()
                accuracy = correct/float(batch_size*num_points)
                iou = compute_iou(targets, pred_choice)

                # update epoch loss and accuracy
                _test_loss.append(loss.item())
                _test_accuracy.append(accuracy)
                _test_iou.append(iou.item())
            
            test_loss.append(np.mean(_test_loss))
            test_accuracy.append(np.mean(_test_accuracy))
            test_iou.append(np.mean(_test_iou))
            print(f'Epoch: {epoch + 1} - Valid Loss: {test_loss[-1]:.4f} ' \
                + f'- Valid Accuracy: {test_accuracy[-1]:.4f} ' \
                + f'- Valid IOU: {test_iou[-1]:.4f}')
            
            # Save best models
            if (test_iou[-1] >= best_iou):
                best_iou = test_iou[-1]
                best_epoch = epoch + 1
                savepath = str(exp_dir) + '/best_model.pth'
                log_string('Saving model at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'train_iou': train_iou,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_iou': test_iou,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_type': args.optimizer,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            global_epoch += 1            

    log_string('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
