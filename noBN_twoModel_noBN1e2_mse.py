import random
import math
import copy
from datasets import miniImageNet_few_shot, tiered_ImageNet_few_shot, ImageNet_few_shot
from collections import OrderedDict
import warnings
import models
import time
import data
import utils
import sys
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets
import torch.utils.data
from configs import miniImageNet_path, ISIC_path, ChestX_path, CropDisease_path, EuroSAT_path

torch.cuda.empty_cache()


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main(args):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    torch.cuda.empty_cache()
    # Set the scenes
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(
        args.dir, time.strftime("%Y%m%d-%H%M%S") + '_checkpoint.log'), __name__)
    trainlog = utils.savelog(args.dir, 'train')
    vallog = utils.savelog(args.dir, 'val')

    # wandb.init(project='STARTUP',
    #            group=__file__,
    #            name=f'{__file__}_{args.dir}')

    # wandb.config.update(args)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # seed the random number generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ###########################
    # Create Models
    ###########################
    if args.model == 'resnet10':
        backbone = models.ResNet10()
        feature_dim = backbone.final_feat_dim
    elif args.model == 'resnet12':
        backbone = models.Resnet12(width=1, dropout=0.1)
        feature_dim = backbone.output_size
    elif args.model == 'resnet18':
        backbone = models.resnet18(remove_last_relu=False,
                                   input_high_res=True)
        feature_dim = 512
    else:
        raise ValueError('Invalid backbone model')

    backbone_noBN = models.ResNet10_noBN()
    feature_dim_noBN = backbone_noBN.final_feat_dim

    backbone_sd_init = copy.deepcopy(backbone.state_dict())
    backbone_sd_init_noBN = copy.deepcopy(backbone_noBN.state_dict())

    # the student classifier head
    clf = nn.Linear(feature_dim, 1000).to(device)
    clf_noBN = nn.Linear(feature_dim_noBN, 1000).to(device)
    ############################

    ###########################
    # Create DataLoader
    ###########################

    # create the base dataset
    if args.base_dataset == 'miniImageNet':
        base_transform = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        base_transform_test = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)
        if args.base_split is not None:
            base_dataset = miniImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
    elif args.base_dataset == 'tiered_ImageNet':
        if args.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size for is not 84x84")
        base_transform = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_transform_test = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)
        if args.base_split is not None:
            base_dataset = tiered_ImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
    elif args.base_dataset == 'ImageNet':
        if args.base_no_color_jitter:
            base_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            warnings.warn("Using ImageNet with Color Jitter")
            base_transform = ImageNet_few_shot.TransformLoader(
                args.image_size).get_composed_transform(aug=True)
        base_transform_test = ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)

        if args.base_split is not None:
            base_dataset = ImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
        print("Size of Base dataset:", len(base_dataset))
    else:
        raise ValueError("Invalid base dataset!")

    # initialize the student's backbone with random weights
    if args.backbone_random_init:
        backbone.module.load_state_dict(backbone_sd_init)
        backbone_noBN.module.load_state_dict(backbone_sd_init_noBN)

    # Generate trainset and valset for base dataset
    base_ind = torch.randperm(len(base_dataset))

    base_train_ind = base_ind[:int((1 - args.base_val_ratio)*len(base_ind))]
    base_val_ind = base_ind[int((1 - args.base_val_ratio)*len(base_ind)):]

    base_dataset_val = copy.deepcopy(base_dataset)
    base_dataset_val.transform = base_transform_test

    base_trainset = torch.utils.data.Subset(base_dataset, base_train_ind)
    base_valset = torch.utils.data.Subset(base_dataset_val, base_val_ind)

    print("Size of base validation set", len(base_valset))

    base_trainloader = torch.utils.data.DataLoader(base_trainset, batch_size=args.bsize,
                                                   num_workers=args.num_workers,
                                                   shuffle=True, drop_last=True)
    base_valloader = torch.utils.data.DataLoader(base_valset, batch_size=args.bsize * 2,
                                                 num_workers=args.num_workers,
                                                 shuffle=False, drop_last=False)
    ############################

    ###########################
    # Create Optimizer
    ###########################

    optimizer = torch.optim.SGD([
        {'params': filter(lambda p: p.requires_grad, backbone.parameters())},
        {'params': clf.parameters()}
    ],
        lr=0.1, momentum=0.9,
        weight_decay=args.wd,
        nesterov=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min', factor=0.5,
                                                           patience=10, verbose=False,
                                                           cooldown=10,
                                                           threshold_mode='rel',
                                                           threshold=1e-4, min_lr=1e-5)


    optimizer_noBN = torch.optim.SGD([
        {'params': filter(lambda p: p.requires_grad, backbone_noBN.parameters())},
        {'params': clf_noBN.parameters()}
    ],
        lr=0.1, momentum=0.9,
        weight_decay=args.wd,
        nesterov=False)

    scheduler_noBN = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noBN,
                                                           mode='min', factor=0.5,
                                                           patience=10, verbose=False,
                                                           cooldown=10,
                                                           threshold_mode='rel',
                                                           threshold=1e-5, min_lr=1e-6)


    #######################################
    starting_epoch = 0

 
    ###########################################
    ####### Learning rate test ################
    ###########################################

    
    if starting_epoch == 0:
        # Start by doing a learning rate test
        lr_candidates = [1e-1]
        lr_candidates_noBN = [1e-2]

        step = 50

        # number of training epochs to get at least 50 updates
        warm_up_epoch = math.ceil(step / len(base_trainloader))

        # keep track of the student model initialization
        # Need to keep reloading when testing different learning rates
        sd_current = copy.deepcopy(backbone.state_dict())
        sd_head = copy.deepcopy(clf.state_dict())
        
        sd_current_noBN = copy.deepcopy(backbone_noBN.state_dict())
        sd_head_noBN = copy.deepcopy(clf_noBN.state_dict())

        

        # Test the learning rate by training for one epoch
        #vals = lr_test(backbone, clf, sd_current, sd_head, lr_candidates, logger, args, device, base_trainloader, base_valloader, warm_up_epoch)
        
        #vals_noBN = lr_test(backbone_noBN, clf_noBN, sd_current_noBN, sd_head_noBN, lr_candidates_noBN, logger, args, device, base_trainloader, base_valloader, warm_up_epoch)

        



        # pick the best learning rates
        current_lr = lr_candidates[0]
        current_lr_noBN = lr_candidates_noBN[0]

        # reload the models
        backbone.load_state_dict(sd_current)
        clf.load_state_dict(sd_head)

        backbone_noBN.load_state_dict(sd_current_noBN)
        clf_noBN.load_state_dict(sd_head_noBN)

        logger.info(f"** BN Learning with lr: {current_lr}")
        logger.info(f"** noBN Learning with lr: {current_lr_noBN}")
        
        optimizer = torch.optim.SGD([
            {'params': filter(lambda p: p.requires_grad, backbone.parameters())},
            {'params': clf.parameters()}
        ],
            lr=current_lr, momentum=0.9,
            weight_decay=args.wd,
            nesterov=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min', factor=0.5,
                                                               patience=10, verbose=False,
                                                               cooldown=10,
                                                               threshold_mode='rel',
                                                               threshold=1e-4, min_lr=1e-5)

        optimizer_noBN = torch.optim.SGD([
            {'params': filter(lambda p: p.requires_grad, backbone_noBN.parameters())},
            {'params': clf_noBN.parameters()}
        ],
            lr=current_lr_noBN, momentum=0.9,
            weight_decay=args.wd,
            nesterov=False)

        scheduler_noBN = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noBN,
                                                               mode='min', factor=0.5,
                                                               patience=10, verbose=False,
                                                               cooldown=10,
                                                               threshold_mode='rel',
                                                               threshold=1e-5, min_lr=1e-6)

        scheduler.step(math.inf)
        scheduler_noBN.step(math.inf)
        
        best_loss = math.inf
        best_epoch = 0
        checkpoint(backbone_noBN, clf_noBN,
                   optimizer_noBN, scheduler_noBN, os.path.join(
                       args.dir, f'checkpoint_best.pkl'), 0)

    ############################
    # save the initialization
    checkpoint(backbone_noBN, clf_noBN,
               optimizer_noBN, scheduler_noBN,
               os.path.join(
                   args.dir, f'checkpoint_{starting_epoch}.pkl'), starting_epoch)

    try:
        for epoch in tqdm(range(starting_epoch, args.epochs)):
            perf = train(backbone, backbone_noBN, clf, clf_noBN, optimizer, optimizer_noBN,
                         base_trainloader,
                         epoch, args.epochs, logger, trainlog, args, device)

            scheduler.step(perf['Loss/avg'])
            scheduler_noBN.step(perf['Loss_noBN/avg'])

            # Always checkpoint after first epoch of training
            if (epoch == starting_epoch) or ((epoch + 1) % args.save_freq == 0):
                checkpoint(backbone_noBN, clf_noBN,
                           optimizer_noBN, scheduler_noBN,
                           os.path.join(
                               args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)

            if (epoch == starting_epoch) or ((epoch + 1) % args.eval_freq == 0):
                performance_val = validate(backbone_noBN, clf_noBN,
                                           base_valloader,
                                           epoch+1, args.epochs, logger, vallog, args, device, postfix='Validation')

                loss_val = performance_val['Loss_test/avg']

                if best_loss > loss_val:
                    best_epoch = epoch + 1
                    checkpoint(backbone_noBN, clf_noBN,
                               optimizer_noBN, scheduler_noBN, os.path.join(
                                   args.dir, f'checkpoint_best.pkl'), best_epoch)
                    logger.info(
                        f"*** Best model checkpointed at Epoch {best_epoch}")
                    best_loss = loss_val

        if (epoch + 1) % args.save_freq != 0:
            checkpoint(backbone_noBN, clf_noBN,
                       optimizer_noBN, scheduler_noBN, os.path.join(
                           args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)
    finally:
        trainlog.save()
        vallog.save()
    return


def checkpoint(model, clf, optimizer, scheduler, save_path, epoch):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    sd = {
        'model': copy.deepcopy(model.state_dict()),
        'clf': copy.deepcopy(clf.state_dict()),
        'opt': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'epoch': epoch
    }

    torch.save(sd, save_path)
    return sd


def load_checkpoint(model, clf, optimizer, scheduler, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])
    clf.load_state_dict(sd['clf'])
    optimizer.load_state_dict(sd['opt'])
    scheduler.load_state_dict(sd['scheduler'])

    return sd['epoch']


def train(model, model_noBN, clf, clf_noBN,
          optimizer, optimizer_noBN, base_trainloader, epoch,
          num_epochs, logger, trainlog, args, device, turn_off_sync=False):

    meters = utils.AverageMeterSet()
    model.to(device)
    model.train()
    clf.train()
    model_noBN.to(device)
    model_noBN.train()
    clf_noBN.train()

    mse_criterion = nn.MSELoss()
    mse_noBN = nn.MSELoss()
    loss_ce = nn.CrossEntropyLoss()
    loss_ce_noBN = nn.CrossEntropyLoss()

    end = time.time()
    for i, (X_base, y_base) in enumerate(base_trainloader):

        meters.update('Data_time', time.time() - end)

        current_lr = optimizer.param_groups[0]['lr']
        current_lr_noBN = optimizer_noBN.param_groups[0]['lr']
        meters.update('lr', current_lr_noBN, 1)

        X_base = X_base.to(device)
        y_base = y_base.to(device)

        #optimizer.zero_grad()
        #model_bn = copy.deepcopy

        features_base = model(X_base)
        logits_base = clf(features_base)

        loss_base = loss_ce(logits_base, y_base)

        loss = loss_base

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        #optimizer_noBN.zero_grad()

        features_base_noBN = model_noBN(X_base)
        logits_base_noBN = clf_noBN(features_base_noBN)
        
        #logits_bn = clf(model(X_base))
        loss_base_noBN = loss_ce(logits_base_noBN, y_base)
        loss_diff = mse_criterion(logits_base_noBN, logits_base.detach())
        loss_noBN = loss_base_noBN + loss_diff

        loss_noBN.backward()
        optimizer_noBN.step()
        optimizer_noBN.zero_grad()


        #.deepcopy()#######################################
        '''
        print("14")
        optimizer_bn.zero_grad()
        
        features_base = model(X_base)
        logits_base = clf(features_base)    

        loss_base = loss_ce(logits_base, y_base)
        loss_bn = loss_base


        print("15")
        loss_bn.backward()
        optimizer.step()

        print("16")

        optimizer_noBN.zero_grad()

        features_base_noBN = model_noBN(X_base)
        logits_base_fn = clf_fn(features_base_fn)

        loss_base_noBN = loss_ce(logits_base_noBN, y_base)
        loss_diff = mse_criterion(logits_base_noBN, logits_base_bn.detach())
        loss_noBN = loss_base_noBN + loss_diff

        loss_noBN.backward()
        optimizer_noBN.step()
        '''
        ########################################

        meters.update('Loss', loss.item(), 1)
        meters.update('Loss_noBN', loss_noBN.item(), 1)        
        meters.update('CE_Loss_source', loss_base.item(), 1)
        meters.update('CE_Loss_source_noBN', loss_base_noBN.item(), 1)

        perf_base = utils.accuracy(logits_base.data,
                                   y_base.data, topk=(1, ))
        perf_base_noBN = utils.accuracy(logits_base_noBN.data,
                                   y_base.data, topk=(1, ))
        meters.update('top1_base', perf_base['average'][0].item(), len(X_base))
        meters.update('top1_base_per_class',
                      perf_base['per_class_average'][0].item(), 1)
        meters.update('top1_base_noBN', perf_base_noBN['average'][0].item(), len(X_base))
        meters.update('top1_base_per_class_noBN',
                      perf_base_noBN['per_class_average'][0].item(), 1)

        meters.update('Batch_time', time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            values = meters.values()
            averages = meters.averages()
            sums = meters.sums()

            logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step} / {steps}] '
                             'Batch Time: {meters[Batch_time]:.4f} '
                             'Data Time: {meters[Data_time]:.4f} '
                             'Average Loss: {meters[Loss]:.4f} Average Loss noBN: {meters[Loss_noBN]:.4f}'
                             'Average CE Loss (Source): {meters[CE_Loss_source]: .4f} '
                             'Average CE Loss noBN (Source): {meters[CE_Loss_source_noBN]: .4f} '
                             'Learning Rate: {meters[lr]:.4f} '
                             'Top1_base: {meters[top1_base]:.4f} '
                             'Top1_base_per_class: {meters[top1_base_per_class]:.4f} '
                             'Top1_base_noBN: {meters[top1_base_noBN]:.4f} '
                             'Top1_base_per_class_noBN: {meters[top1_base_per_class_noBN]:.4f} '
                             ).format(
                epoch=epoch, epochs=num_epochs, step=i+1, steps=len(base_trainloader), meters=meters)

            logger.info(logger_string)
            print(logger_string)

        if (args.iteration_bp is not None) and (i+1) == args.iteration_bp:
            break

    logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step}] Batch Time: {meters[Batch_time]:.4f} '
                     'Data Time: {meters[Data_time]:.4f} '
                     'Average Loss: {meters[Loss]:.4f} Average Loss noBN: {meters[Loss_noBN]:.4f}'
                                                  'Average CE Loss (Source): {meters[CE_Loss_source]: .4f} '
                     'Average CE Loss noBN (Source): {meters[CE_Loss_source_noBN]: .4f} '
                     'Learning Rate: {meters[lr]:.4f} '
                     'Top1_base: {meters[top1_base]:.4f} '
                     'Top1_base_per_class: {meters[top1_base_per_class]:.4f} '
                     'Top1_base_noBN: {meters[top1_base_noBN]:.4f} '
                     'Top1_base_per_class_noBN: {meters[top1_base_per_class_noBN]:.4f} '
                     ).format(
        epoch=epoch+1, epochs=num_epochs, step=0, meters=meters)

    logger.info(logger_string)
    print(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    trainlog.record(epoch+1, {
        **values,
        **averages,
        **sums
    })

    return averages


def validate(model, clf,
             base_loader, epoch, num_epochs, logger,
             testlog, args, device, postfix='Validation', turn_off_sync=False):
    meters = utils.AverageMeterSet()
    model.to(device)
    model.eval()
    clf.eval()

    loss_ce = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    end = time.time()

    logits_base_all = []
    ys_base_all = []
    with torch.no_grad():
        # Compute the loss on the source base dataset
        for X_base, y_base in base_loader:
            X_base = X_base.to(device)
            y_base = y_base.to(device)

            features = model(X_base)
            logits_base = clf(features)

            logits_base_all.append(logits_base)
            ys_base_all.append(y_base)

    ys_base_all = torch.cat(ys_base_all, dim=0)
    logits_base_all = torch.cat(logits_base_all, dim=0)

    loss_base = loss_ce(logits_base_all, ys_base_all)

    loss = loss_base

    meters.update('CE_Loss_source_test', loss_base.item(), 1)
    meters.update('Loss_test', loss.item(), 1)

    perf_base = utils.accuracy(logits_base_all.data,
                               ys_base_all.data, topk=(1, ))

    meters.update('top1_base_test', perf_base['average'][0].item(), 1)
    meters.update('top1_base_test_per_class',
                  perf_base['per_class_average'][0].item(), 1)

    meters.update('Batch_time', time.time() - end)

    logger_string = ('{postfix} Epoch: [{epoch}/{epochs}]  Batch Time: {meters[Batch_time]:.4f} '
                     'Average Test Loss: {meters[Loss_test]:.4f} '
                     'Average CE Loss (Source): {meters[CE_Loss_source_test]: .4f} '
                     'Top1_base_test: {meters[top1_base_test]:.4f} '
                     'Top1_base_test_per_class: {meters[top1_base_test_per_class]:.4f} ').format(
        postfix=postfix, epoch=epoch, epochs=num_epochs, meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    testlog.record(epoch, {
        **values,
        **averages,
        **sums
    })

    if postfix != '':
        postfix = '_' + postfix

    return averages


            


def lr_test(backbone, backbone_noBN, clf, clf_noBN, sd_current, sd_head, 
            sd_current_noBN, sd_head_noBN, lr_candidates, lr_candidates_noBN, 
            logger, args, device, base_trainloader, base_valloader, warm_up_epoch):

    vals = []    
    for current_lr in lr_candidates:
        lr_log = utils.savelog(args.dir, f'lr_bn_{current_lr}')
        lr_log = utils.savelog(args.dir, f'lr_nobn_{current_lr}')

        # reload the student model
        backbone.load_state_dict(sd_current)
        clf.load_state_dict(sd_head)

        # create the optimizer
        optimizer = torch.optim.SGD([
            {'params': filter(lambda p: p.requires_grad, backbone.parameters())},
            {'params': clf.parameters()}
        ],            
            lr=current_lr, momentum=0.9,
            weight_decay=args.wd,
            nesterov=False)

        logger.info(f'*** Testing Learning Rate: {current_lr}')

        # training for a bit
        for i in range(warm_up_epoch):
            perf = train(backbone, clf, optimizer,
                        base_trainloader,
                        i, warm_up_epoch, logger, lr_log, args, device, turn_off_sync=True)

        # compute the validation loss for picking learning rates
        perf_val = validate(backbone, clf,
                            base_valloader,
                            1, 1, logger, vallog, args, device, postfix='Validation',
                            turn_off_sync=True)
        vals.append(perf_val['Loss_test/avg'])    
    return vals       


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='STARTUP')
    parser.add_argument('--dir', type=str, default='./logs/baseline_na/',
                        help='directory to save the checkpoints')

    parser.add_argument('--bsize', type=int, default=32,
                        help='batch_size for STARTUP')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency (in epoch) to save')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='Frequency (in epoch) to evaluate on the val set')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Frequency (in step per epoch) to print training stats')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay for the model')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')

    parser.add_argument('--iteration_bp', type=int,
                        help='which step to break in the training loop')
    parser.add_argument('--model', type=str, default='resnet10',
                        help='Backbone model')

    parser.add_argument('--backbone_random_init', action='store_true',
                        help="Use random initialized backbone ")

    parser.add_argument('--base_dataset', type=str,
                        default='miniImageNet', help='base_dataset to use')
    parser.add_argument('--base_path', type=str,
                        default=miniImageNet_path, help='path to base dataset')
    parser.add_argument('--base_split', type=str,
                        help='split for the base dataset')
    parser.add_argument('--base_no_color_jitter', action='store_true',
                        help='remove color jitter for ImageNet')
    parser.add_argument('--base_val_ratio', type=float, default=0.05,
                        help='amount of base dataset set aside for validation')

    parser.add_argument('--batch_validate', action='store_true',
                        help='to do batch validate rather than validate on the full dataset (Ideally, for SimCLR,' +
                        ' the validation should be on the full dataset but might not be feasible due to hardware constraints')

    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')

    args = parser.parse_args()
    main(args)
