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
import yaml
import datetime

from torch.utils.data import DataLoader

#import data_utils.DataAugmentationAndShuffle as DataAugmentator
from data_utils.metrics import compute_iou
from data_utils.s3_dis_dataset import S3DIS
from data_utils.dales_dataset import DalesDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_modules_dict = {'pointnet_sem_seg': 'models.point_net_sem_segmentation',
                       'pointnet_v2_sem_seg_ssg': 'models.point_net_v2_sem_segmentation_ssg'}


'''hparams_for_args_default = {
    'num_point': 8192,
    'batch_size': 8,
    'dropout': 0.4,
    'extra_feat_dropout': 0.2,
    'label_smoothing': 0.0,
    'optimizer': 'AdamW',  
    'learning_rate': 1e-3,
    'scheduler': 'Cosine'  
}'''

# hparams for arguments to evaluate model performance
hparams_for_args_to_evaluate = {
    'num_point': 8192,          #4096, 8192, 16384, 32768
    # one run with --use_extra_features
    ## one run with --use_fps
    'batch_size': 12,            #8, 16, 32, 64
    'dropout': 0.5,             #0.0, 0.2, 0.5
    'extra_feat_dropout': 0.2,  #0.0, 0.2, 0.5
    'label_smoothing': 0.1,     #0.0, 0.1, 0.2
    'optimizer': 'AdamW',       #AdamW, Adam, SGD
    'learning_rate': 1e-3,      #1e-2, 1e-3, 1e-4
    'scheduler': 'Cosine'       #Cosine, Cyclic, Step
}


# =========================================================================================================================================================
# ==================================================   ARGUMENT PARSER METHOD   ===========================================================================
# =========================================================================================================================================================
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training_cls')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    #parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # Modelnet data loader arguments
    parser.add_argument('--num_point', type=int, default=hparams_for_args_to_evaluate['num_point'], help='Point Number')
    parser.add_argument('--train_area', type=int, nargs='+', default=[1, 2, 4], help='Which area to use for test, option: 1-3 [default: [1,2,4]]')
    parser.add_argument('--test_area', type=int, nargs='+', default=[3], help='Which area to use for test, option: 1-4 [default: [3]]')
    parser.add_argument('--use_extra_features', action='store_true', default=False, help='use extra features as RGB info or more')
    # TO DO: FPS
    #parser.add_argument('--use_fps', action='store_true', default=False, help='use further point sampiling')
    # TO DO: DATA PREPROCESSING
    #parser.add_argument('--no_data_preprocess', action='store_true', default=False, help='preprocess the data or process it during the getitem call')
    # Dataset selection
    parser.add_argument('--dataset', default='s3dis', help='dataset name, options are [s3dis, dales]')
    parser.add_argument('--dataset_path', default='data/stanford_indoor3d', help='Path to the dataset [default: data/stanford_indoor3d]')
    # Specific parameters for dales dataset
    parser.add_argument('--partitions', type=int, default=10, help='Number of partitions to split the data')
    parser.add_argument('--overlap', type=float, default=0.1, help='Overlap between partitions')
    # Model selection
    parser.add_argument('--model', default='pointnet_v2_sem_seg_ssg', help='model name [default: pointnet_sem_seg]')
    # Model parameters
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--batch_size', type=int, default=hparams_for_args_to_evaluate['batch_size'], help='batch size in training')
    parser.add_argument('--dropout', type=float, default=hparams_for_args_to_evaluate['dropout'], help='Dropout')
    parser.add_argument('--extra_feat_dropout', type=float, default=hparams_for_args_to_evaluate['extra_feat_dropout'], help='Extra Features Dropout to avoid the classifier rely on them')
    parser.add_argument('--label_smoothing', type=float, default=hparams_for_args_to_evaluate['label_smoothing'], help='Loss label smoothing used for the cross entropy')
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default=hparams_for_args_to_evaluate['optimizer'], help='optimizer for training [AdamW, Adam, SGD]')
    parser.add_argument('--learning_rate', type=float, default=hparams_for_args_to_evaluate['learning_rate'], help='learning rate in training')
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default=hparams_for_args_to_evaluate['scheduler'], help='scheduler for training [Cosine (CosineAnnealingLR), Cyclic (CyclicLR), Step (StepLR)]')
    # Other logging parameters
    parser.add_argument('--remove_checkpoint', action='store_true', default=False, help='remove last checkpoint train progress')
    parser.add_argument('--use_mlflow', action='store_true', default=False, help='Log train with mlflow')
    parser.add_argument('--mlflow_run_name', type=str, default='pointnet_sem_segmentation', help='Name of the mlflow run')
    return parser.parse_args()

def dump_args_to_yaml(args, yaml_file):
    args_dict = vars(args)  # Convert Namespace to dictionary
    with open(yaml_file, 'w') as file:
        yaml.dump(args_dict, file)

# =========================================================================================================================================================
# ==================================================   MAIN METHOD CONTAINING TRAINING   ==================================================================
# =========================================================================================================================================================
def main(args):
    # ===============================================================
    # CHECKPOINT DIRECTORY CHECKS
    # ===============================================================
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('semantic_seg')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.model)
    if args.remove_checkpoint and exp_dir.exists():
        print('Deleting existing checkpoint!')
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(exist_ok=True)
    
    # ===============================================================
    # PARAMETERS SELECTED
    # ===============================================================
    print('PARAMETERS:')
    print(args)
    num_points = args.num_point
    batch_size = args.batch_size

    # ===============================================================
    # DATA LOADING
    # ===============================================================
    print('Loading dataset...')
    data_path = args.dataset_path
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")

    train_dataset = None
    eval_dataset = None
    if args.dataset == 's3dis':
        train_dataset = S3DIS(root=data_path, area_nums=args.train_area, split='train', npoints=num_points, r_prob=0.25, include_rgb=args.use_extra_features)
        eval_dataset = S3DIS(root=data_path, area_nums=args.test_area, split='test', npoints=num_points, r_prob=0.25, include_rgb=args.use_extra_features)
    elif args.dataset == 'dales':
        train_dataset = DalesDataset(root=data_path, split='train', partitions=args.partitions, overlap=args.overlap, npoints=num_points)
        eval_dataset = DalesDataset(root=data_path, split='test', partitions=args.partitions, overlap=args.overlap, npoints=num_points)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    num_classes = len(train_dataset.get_categories())
    print(f"Number of classes: {num_classes}")

    trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    evalDataLoader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    print("The length of the training data is: %d" % len(train_dataset))
    print("The length of the evaluation data is: %d" % len(eval_dataset))

    # Get an example point to be able to know the input dimension of the data (xyz or xyz + extra features like rgb, normals, etc.)
    example_points, example_target = trainDataLoader.dataset[0]
    input_dimension = example_points.shape[1]

    # ===============================================================
    # MODEL LOADING
    # ===============================================================
    model = importlib.import_module(models_modules_dict[args.model])

    classifier = model.get_model(num_points=num_points, m=num_classes, dropout=args.dropout, input_dim=input_dimension, extra_feat_dropout=args.extra_feat_dropout)
    criterion = model.get_loss(label_smoothing=args.label_smoothing)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # ===============================================================
    # OPTIMIZER SELECTION
    # ===============================================================
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    # ===============================================================
    # SCHEDULER SELECTION
    # ===============================================================
    if args.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch / 10, gamma=0.5)
    
    # ===============================================================
    # MODEL STATE AND METRICS VARIABLES INITIALIZATION
    # ===============================================================
    global_epoch = 0
    start_epoch = 0
    train_loss = []
    train_accuracy = []
    train_iou = []
    eval_loss = []
    eval_accuracy = []
    eval_iou = []
    optim_learning_rate = []
    best_eval_iou = 0.2

    # ===============================================================
    # CHECKPOINT DATA LOADING INTO MODEL AND OPTIMIZER IF AVAILABLE
    # ===============================================================
    try:
        checkpoint = torch.load(str(exp_dir) + '/best_model.pth')
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        train_iou = checkpoint['train_iou']
        eval_loss = checkpoint['eval_loss']
        eval_accuracy = checkpoint['eval_accuracy']
        eval_iou = checkpoint['eval_iou']
        best_eval_iou = eval_iou[-1]
        optim_learning_rate = checkpoint['optim_learning_rate']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_type'] == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    # ===============================================================
    # MLFLOW TRACKING AND LOGGING
    # ===============================================================
    if args.use_mlflow:
        # Export mlflow environment variables
        os.environ['MLFLOW_TRACKING_URI'] = 'http://34.16.143.171:3389'
        os.environ['MLFLOW_EXPERIMENT_NAME'] = 'pointnet_sem_segmentation'
        os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
        run_name = f"{args.mlflow_run_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        mlflow.start_run(run_name=run_name)
        dump_args_to_yaml(args, 'args.yaml')
        # Log the arguments
        mlflow.log_artifact('args.yaml')

    # ===============================================================
    # MODEL TRAINING
    # ===============================================================
    print('Starting training...')
    try:
        for epoch in range(start_epoch, args.epoch):
            # TRAINING
            print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
            classifier = classifier.train()
            train_instance_loss = []
            train_instance_accuracy = []
            train_instance_iou = []

            for i, (points, targets) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                
                if not args.use_cpu:
                    points, targets = points.transpose(2, 1).cuda(), targets.cuda()
                else:
                    points, targets = points.transpose(2, 1), targets
                
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
                
                # get metrics
                correct = pred_choice.eq(targets.data).cpu().sum()
                accuracy = correct/float(batch_size*num_points)
                iou = compute_iou(targets, pred_choice)

                train_instance_loss.append(loss.item())
                train_instance_accuracy.append(accuracy)
                train_instance_iou.append(iou.item())
            
            # update epoch loss and accuracy
            train_epoch_loss = np.mean(train_instance_loss)
            train_epoch_accuracy = np.mean(train_instance_accuracy)
            train_epoch_iou = np.mean(train_instance_iou)

            current_lr = scheduler.get_last_lr()[0]
            optim_learning_rate.append(current_lr)
            scheduler.step()

            if args.use_mlflow:
                mlflow.log_metric('train_loss', train_epoch_loss, step=epoch)
                mlflow.log_metric('train_accuracy', train_epoch_accuracy, step=epoch)
                mlflow.log_metric('train_iou', train_epoch_iou, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            train_loss.append(train_epoch_loss)
            train_accuracy.append(train_epoch_accuracy)
            train_iou.append(train_epoch_iou)

            print(f'Epoch: {epoch + 1} - Train Loss: {train_loss[-1]:.4f} ' \
                + f'- Train Accuracy: {train_accuracy[-1]:.4f} ' \
                + f'- Train IOU: {train_iou[-1]:.4f}')

            
            # EVALUATION
            with torch.no_grad():
                eval_epoch_loss, eval_epoch_acc, eval_epoch_iou = evaluate_model(classifier.eval(), criterion, evalDataLoader, batch_size, num_points)

                if args.use_mlflow:
                    mlflow.log_metric('eval_loss', eval_epoch_loss, step=epoch)
                    mlflow.log_metric('eval_accuracy', eval_epoch_acc, step=epoch)
                    mlflow.log_metric('eval_iou', eval_epoch_iou, step=epoch)

                # Save the epoch evaluation metrics
                eval_loss.append(eval_epoch_loss)
                eval_accuracy.append(eval_epoch_acc)
                eval_iou.append(eval_epoch_iou)

                # Print evaluation results to keep track of the improvements
                print(f'Epoch: {epoch + 1} - Valid Loss: {eval_loss[-1]:.4f} ' \
                    + f'- Valid Accuracy: {eval_accuracy[-1]:.4f} ' \
                    + f'- Valid IOU: {eval_iou[-1]:.4f}')
                
                # Save a checkpoint with all the relevant status info it the model has improved
                if (eval_iou[-1] >= best_eval_iou):
                    best_eval_iou = eval_iou[-1]
                    best_epoch = epoch + 1
                    savepath = str(exp_dir) + '/best_model.pth'
                    print('Saving model at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'train_iou': train_iou,
                        'optim_learning_rate': optim_learning_rate,
                        'eval_loss': eval_loss,
                        'eval_accuracy': eval_accuracy,
                        'eval_iou': eval_iou,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_type': args.optimizer,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    #if args.use_mlflow:
                    #    mlflow.log_artifact(savepath)

                # Next epoch
                global_epoch += 1

    finally:
        # End the MLflow run if use_mlflow is True
        if args.use_mlflow:
            mlflow.end_run()

    print('Training completed!')


# =========================================================================================================================================================
# ==================================================   EVALUATION METHOD TO USE IN TRAINING   =============================================================
# =========================================================================================================================================================
def evaluate_model(model, criterion, loader, batch_size, num_points):
    eval_instance_loss = []
    eval_instance_accuracy = []
    eval_instance_iou = []
    
    eval_classifier = model.eval()

    for i, (points, targets) in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):
        
        if not args.use_cpu:
            points, targets = points.transpose(2, 1).cuda(), targets.squeeze().cuda()
        else:
            points, targets = points.transpose(2, 1), targets.squeeze()

        preds, crit_idxs, feat_trans = eval_classifier(points)
        pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

        loss = criterion(preds, targets, pred_choice) 

        # get metrics
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct/float(batch_size*num_points)
        iou = compute_iou(targets, pred_choice)

        # update epoch loss and accuracy
        eval_instance_loss.append(loss.item())
        eval_instance_accuracy.append(accuracy)
        eval_instance_iou.append(iou.item())
    
    eval_epoch_loss_ = np.mean(eval_instance_loss)
    eval_epoch_accuracy_ = np.mean(eval_instance_accuracy)
    eval_epoch_iou_ = np.mean(eval_instance_iou)

    return eval_epoch_loss_, eval_epoch_accuracy_, eval_epoch_iou_

# =========================================================================================================================================================
# ==================================================   MAIN   =============================================================================================
# =========================================================================================================================================================
if __name__ == '__main__':
    args = parse_args()
    main(args)
