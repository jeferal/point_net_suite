import os
import sys
import torch
import numpy as np
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import mlflow

from torch.utils.data import DataLoader
from point_net_suite.data_utils.metrics import compute_iou
from point_net_suite.data_utils.model_net import ModelNetDataLoader
import point_net_suite.data_utils.augmentation as DataAugmentator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_folder_dict = {'pointnet_cls': 'models/Pointnet'}
models_modules_dict = {'pointnet_cls': 'point_net_suite.models.point_net_classification'}

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training_cls')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    #parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training')
    #parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--remove_checkpoint', action='store_true', default=False, help='remove last checkpoint train progress')
    parser.add_argument('--use_mlflow', action='store_true', default=False, help='Log train with mlflow')
    return parser.parse_args()


def evaluate_model(model, criterion, loader, num_class=40):
    # Matrix to store the per class accuracy
    class_acc = np.zeros((num_class, 3)) #dim0 = accumulated sum of accuracies, dim1 = number of batches, dim2 = accuracy per class calculated in the end of the loop (dim0 / dim1)
    # Array to store the batch accuracy (ratio of total correct predictions per batch)
    _test_loss = []
    _mean_acc = []
    _test_iou = []
    
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, crit_idxs, feat_trans = classifier(points)
        loss = criterion(pred, target.long(), feat_trans) 

        # Chooses the class with the highest probability for each point in the batch (pred is a matrix of shape = (num_points in bactch, num_clases) where the second dimension is the probability predicted for each class)
        pred_choice = pred.data.max(1)[1]        

        # Class accuracy calculation
        for curr_class in np.unique(target.cpu()):
            # Compares the predictions masked with the current class, effectively counting the number of correct predictions for the current class with sum
            curr_cat_classacc = pred_choice[target == curr_class].eq(target[target == curr_class].long().data).cpu().sum()
            # Computes and accumulates the accuracy for the class in the current batch (number of correct predictions divided by the number of samples for the class)
            class_acc[curr_class, 0] += curr_cat_classacc.item() / float(points[target == curr_class].size()[0])
            # Adds one more batch count to the class (so we can later get the mean class accuracy - we will have the accumulation of accuracy and the number of batches to divide the accumulation by)
            class_acc[curr_class, 1] += 1 

        # Batch metrics calculation
        _test_loss.append(loss.item())
        correct = pred_choice.eq(target.long().data).cpu().sum()
        _mean_acc.append(correct.item() / float(points.size()[0]))
        iou = compute_iou(target, pred_choice)
        _test_iou.append(iou.item())

    # Loss mean
    test_loss_t = np.mean(_test_loss)

    # Get the accuracy per class by dividing the accumulated accuracy by the batch count and store it in dim2
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # Get the mean accuracy for all classes
    mean_class_acc_t = np.mean(class_acc[:, 2])

    # Calculate the mean of the batches accuracy
    instance_acc_t = np.mean(_mean_acc)

    # Calculate the mean iou
    instance_iou_t = np.mean(_test_iou)

    return test_loss_t, instance_acc_t, mean_class_acc_t, instance_iou_t


def main(args):
    def log_string(str):
        #logger.info(str)
        print(str)

    #'''HYPER PARAMETER'''
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.model)
    if args.remove_checkpoint and exp_dir.exists():
        log_string('Deleting existing checkpoint!')
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(exist_ok=True)

    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    '''MODEL LOADING'''
    sys.path.append(os.path.join(BASE_DIR, models_folder_dict[args.model]))
    num_class = args.num_category
    model = importlib.import_module(models_modules_dict[args.model])

    classifier = model.get_model(num_points=args.num_point, k=num_class, dropout=args.dropout)
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    train_loss = []
    train_accuracy = []
    train_iou = []
    test_loss = []
    test_accuracy = []
    test_mean_class_accuracy = []
    test_iou = []
    best_instance_acc = 0.0
    best_class_acc = 0.0

    try:
        checkpoint = torch.load(str(exp_dir) + '/best_model.pth')
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        train_iou = checkpoint['train_iou']
        test_loss = checkpoint['test_loss']
        test_accuracy = checkpoint['test_accuracy']
        test_mean_class_accuracy = checkpoint['test_mean_class_accuracy']
        test_iou = checkpoint['test_iou']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_type'] == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    
    global_epoch = 0
    global_step = 0

    '''TRANING'''
    log_string('Start training...')
    if args.use_mlflow:
        # Export mlflow environment variables
        os.environ['MLFLOW_TRACKING_URI'] = 'http://34.16.143.171:3389'
        os.environ['MLFLOW_EXPERIMENT_NAME'] = 'pointnet_classification'
        os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
        mlflow.start_run()

    try:
        # Train
        for epoch in range(start_epoch, args.epoch):
            log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
            _train_loss = []
            _mean_acc = []
            _train_iou = []
            classifier = classifier.train()

            scheduler.step()
            for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                optimizer.zero_grad()

                points = points.data.numpy()
                points = DataAugmentator.random_point_dropout(points)
                points[:, :, 0:3] = DataAugmentator.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = DataAugmentator.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                points = points.transpose(2, 1)

                if not args.use_cpu:
                    points, target = points.cuda(), target.cuda()

                pred, crit_idxs, feat_trans = classifier(points)

                loss = criterion(pred, target.long(), feat_trans)
                pred_choice = pred.data.max(1)[1]
                
                loss.backward()
                optimizer.step()

                _train_loss.append(loss.item())

                correct = pred_choice.eq(target.long().data).cpu().sum()
                _mean_acc.append(correct.item() / float(points.size()[0]))

                iou = compute_iou(target, pred_choice)
                _train_iou.append(iou.item())

                global_step += 1


            mean_train_loss = np.mean(_train_loss)
            mean_train_acc = np.mean(_mean_acc)
            mean_train_iou = np.mean(_train_iou)

            mlflow.log_metric("train_loss", mean_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", mean_train_acc, step=epoch)
            mlflow.log_metric("train_iou", mean_train_iou, step=epoch)

            train_loss.append(mean_train_loss)
            train_instance_acc = mean_train_acc
            log_string('Train Instance Accuracy: %f' % train_instance_acc)
            train_accuracy.append(train_instance_acc)
            train_iou.append(mean_train_iou)
            
            # Test
            with torch.no_grad():
                instance_loss, instance_acc, mean_class_acc, instance_iou = evaluate_model(classifier.eval(), criterion, testDataLoader, num_class=num_class)

                mlflow.log_metric("test_loss", instance_loss, step=epoch)
                mlflow.log_metric("test_accuracy", instance_acc, step=epoch)
                mlflow.log_metric("test_mean_class_accuracy", mean_class_acc, step=epoch)
                mlflow.log_metric("test_iou", instance_iou, step=epoch)

                test_loss.append(instance_loss)
                test_accuracy.append(instance_acc)
                test_mean_class_accuracy.append(mean_class_acc)
                test_iou.append(instance_iou)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (mean_class_acc >= best_class_acc):
                    best_class_acc = mean_class_acc
                log_string('Test Instance Accuracy: %f, Mean Class Accuracy: %f' % (instance_acc, mean_class_acc))
                log_string('Best Instance Accuracy: %f, Mean Class Accuracy: %f' % (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc):
                    savepath = str(exp_dir) + '/best_model.pth'
                    log_string('Saving model at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'train_iou': train_iou,
                        'test_loss': test_loss,
                        'test_accuracy': test_accuracy,
                        'test_mean_class_accuracy': test_mean_class_accuracy,
                        'test_iou': test_iou,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_type': args.optimizer,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    mlflow.log_artifact(savepath)
                global_epoch += 1
    finally:
        # End the MLflow run if use_mlflow is True
        if args.use_mlflow:
            mlflow.end_run()

    log_string('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
