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
from data_utils.model_net import ModelNetDataLoader
import data_utils.augmentation as DataAugmentator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_modules_dict = {'pointnet_cls': 'models.point_net_classification',
                       'pointnet_v2_cls_ssg': 'models.point_net_v2_classification_ssg',
                       'pointnet_v2_cls_msg': 'models.point_net_v2_classification_msg'}


'''hparams_for_args_default = {
    'num_point': 1024,
    'batch_size': 24,
    'dropout': 0.4,
    'extra_feat_dropout': 0.2,
    'label_smoothing': 0.0,
    'optimizer': 'AdamW',  
    'learning_rate': 1e-3,
    'scheduler': 'Cosine'  
}'''

# hparams for arguments to evaluate model performance
hparams_for_args_to_evaluate = {
    'num_point': 1024,          #1024, 2048
    # one run with --use_extra_features
    # one run with --use_fps
    'batch_size': 16,           #12, 24, 48, 96 
    'dropout': 0.5,                #0.0, 0.2, 0.5
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
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=hparams_for_args_to_evaluate['num_point'], help='Point Number')
    parser.add_argument('--use_extra_features', action='store_true', default=False, help='use extra features as RGB info or more')
    parser.add_argument('--use_fps', action='store_true', default=False, help='use further point sampiling')
    parser.add_argument('--no_data_preprocess', action='store_true', default=False, help='preprocess the data or process it during the getitem call')
    # Model selection
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
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
    return parser.parse_args()


# =========================================================================================================================================================
# ==================================================   MAIN METHOD CONTAINING TRAINING   ==================================================================
# =========================================================================================================================================================
def main(args):
    # ===============================================================
    # CHECKPOINT DIRECTORY CHECKS
    # ===============================================================
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
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

    # ===============================================================
    # DATA LOADING
    # ===============================================================
    print('Loading dataset...')
    data_path = 'data/modelnet40/'
    data_preprocess = not args.no_data_preprocess
    
    train_dataset = ModelNetDataLoader(root=data_path, split='train', num_cat=args.num_category, num_point=args.num_point,
                                       use_extra_feat=args.use_extra_features, use_fps=args.use_fps, pre_process_data=data_preprocess)
    eval_dataset = ModelNetDataLoader(root=data_path, split='test', num_cat=args.num_category, num_point=args.num_point,
                                      use_extra_feat=args.use_extra_features, use_fps=args.use_fps, pre_process_data=data_preprocess)
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    evalDataLoader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Get an example point to be able to know the input dimension of the data (xyz or xyz + extra features like rgb, normals, etc.)
    example_points, example_target = trainDataLoader.dataset[0]
    input_dimension = example_points.shape[1]

    # ===============================================================
    # MODEL LOADING
    # ===============================================================
    print('Loading selected model...')
    num_class = args.num_category
    model = importlib.import_module(models_modules_dict[args.model])

    classifier = model.get_model(num_points=args.num_point, k=num_class, dropout=args.dropout, input_dim=input_dimension, extra_feat_dropout=args.extra_feat_dropout)
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
    eval_loss = []
    eval_accuracy = []
    eval_mean_class_accuracy = []
    optim_learning_rate = []
    best_eval_acc = 0.0
    best_eval_class_acc = 0.0
    
    # ===============================================================
    # CHECKPOINT DATA LOADING INTO MODEL AND OPTIMIZER IF AVAILABLE
    # ===============================================================
    try:
        checkpoint = torch.load(str(exp_dir) + '/best_model.pth')
        start_epoch = checkpoint['epoch']
        best_eval_acc = checkpoint['best_accuracy']
        best_eval_class_acc = checkpoint['best_class_accuracy']
        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        eval_loss = checkpoint['eval_loss']
        eval_accuracy = checkpoint['eval_accuracy']
        eval_mean_class_accuracy = checkpoint['eval_mean_class_accuracy']
        optim_learning_rate = checkpoint['optim_learning_rate']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_type'] == args.optimizer and checkpoint['scheduler_type'] == args.scheduler:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print('Not loading optimizer and scheduler because the saved states belong to a different type.')
        print('Using pretrained model')
    except:
        print('No existing checkpoint for model, starting training from scratch...')
        start_epoch = 0

    # ===============================================================
    # MLFLOW TRACKING AND LOGGING
    # ===============================================================
    if args.use_mlflow:
        # Export mlflow environment variables
        os.environ['MLFLOW_TRACKING_URI'] = 'http://34.16.143.171:3389'
        os.environ['MLFLOW_EXPERIMENT_NAME'] = 'pointnet_classification'
        os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
        mlflow.start_run()
    

    # ===============================================================
    # MODEL TRAINING
    # ===============================================================
    print('Starting training...')
    try:
        for epoch in range(start_epoch, args.epoch):
            # TRAINING
            print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
            train_instance_loss = []
            train_instance_acc = []
            classifier = classifier.train()
            
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

                train_instance_loss.append(loss.item())

                correct = pred_choice.eq(target.long().data).cpu().sum()
                train_instance_acc.append(correct.item() / float(points.size()[0]))

            train_epoch_loss = np.mean(train_instance_loss)
            train_epoch_acc = np.mean(train_instance_acc)

            current_lr = scheduler.get_last_lr()[0]
            optim_learning_rate.append(current_lr)
            scheduler.step()

            if args.use_mlflow:
                mlflow.log_metric("train_loss", train_epoch_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_epoch_acc, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            train_loss.append(train_epoch_loss)
            train_instance_acc = train_epoch_acc
            print('Train Instance Accuracy: %f' % train_instance_acc)
            train_accuracy.append(train_instance_acc)
            
            # EVALUATING
            with torch.no_grad():
                eval_epoch_loss, eval_epoch_acc, eval_epoch_mean_class_acc, eval_epoch_loss_per_class = evaluate_model(classifier.eval(), criterion, evalDataLoader, num_class=num_class)

                if args.use_mlflow:
                    mlflow.log_metric("eval_loss", eval_epoch_loss, step=epoch)
                    mlflow.log_metric("eval_accuracy", eval_epoch_acc, step=epoch)
                    mlflow.log_metric("eval_mean_class_accuracy", eval_epoch_mean_class_acc, step=epoch)

                     # Log per-class metrics
                    for class_idx in range(num_class):
                        mlflow.log_metric(f"class_{class_idx}_loss", eval_epoch_loss_per_class[class_idx], step=epoch)

                eval_loss.append(eval_epoch_loss)
                eval_accuracy.append(eval_epoch_acc)
                eval_mean_class_accuracy.append(eval_epoch_mean_class_acc)

                # Saving best accuracy values if achieved
                if (eval_epoch_acc >= best_eval_acc):
                    best_eval_acc = eval_epoch_acc
                    best_epoch = epoch + 1

                if (eval_epoch_mean_class_acc >= best_eval_class_acc):
                    best_eval_class_acc = eval_epoch_mean_class_acc

                # Print evaluation results to keep track of the improvements
                print('Eval Instance Accuracy: %f, Mean Class Accuracy: %f' % (eval_epoch_acc, eval_epoch_mean_class_acc))
                print('Best Instance Accuracy: %f, Mean Class Accuracy: %f' % (best_eval_acc, best_eval_class_acc))

                # Save a checkpoint with all the relevant status info it the model has improved
                if (eval_epoch_acc >= best_eval_acc):
                    savepath = str(exp_dir) + '/best_model.pth'
                    print('Saving model at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'best_accuracy': best_eval_acc,
                        'best_class_accuracy': best_eval_class_acc,
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'optim_learning_rate': optim_learning_rate,
                        'eval_loss': eval_loss,
                        'eval_accuracy': eval_accuracy,
                        'eval_mean_class_accuracy': eval_mean_class_accuracy,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_type': args.optimizer,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_type': args.scheduler,
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    torch.save(state, savepath)
                    if args.use_mlflow:
                        mlflow.log_artifact(savepath)

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
def evaluate_model(model, criterion, loader, num_class=40):
    # Matrix to store the per class accuracy
    class_acc = np.zeros((num_class, 3)) #dim0 = accumulated sum of accuracies, dim1 = number of batches, dim2 = accuracy per class calculated in the end of the loop (dim0 / dim1)
    # Array to store the batch accuracy (ratio of total correct predictions per batch)
    eval_instance_loss = []
    eval_instance_mean_acc = []

    # Initialize arrays for class-wise loss and IoU
    class_loss = np.zeros(num_class)
    class_count = np.zeros(num_class)
    
    eval_classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, crit_idxs, feat_trans = eval_classifier(points)
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

            # Accumulate the class-wise loss and count
            class_loss[curr_class] += loss.item()
            class_count[curr_class] += 1

        # Batch metrics calculation
        eval_instance_loss.append(loss.item())
        correct = pred_choice.eq(target.long().data).cpu().sum()
        eval_instance_mean_acc.append(correct.item() / float(points.size()[0]))

    # Loss mean
    eval_epoch_loss_ = np.mean(eval_instance_loss)

    # Calculate the mean of the batches accuracy
    eval_epoch_acc_ = np.mean(eval_instance_mean_acc)

    # Get the accuracy per class by dividing the accumulated accuracy by the batch count and store it in dim2
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # Get the mean accuracy for all classes
    eval_epoch_mean_class_acc_ = np.mean(class_acc[:, 2])

    # Compute mean loss and IoU per class
    eval_epoch_mean_class_loss = class_loss / class_count

    return eval_epoch_loss_, eval_epoch_acc_, eval_epoch_mean_class_acc_, eval_epoch_mean_class_loss


# =========================================================================================================================================================
# ==================================================   MAIN   =============================================================================================
# =========================================================================================================================================================
if __name__ == '__main__':
    args = parse_args()
    main(args)
