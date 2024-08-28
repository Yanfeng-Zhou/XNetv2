from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.backends import cudnn
import random

from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import print_train_loss_XNetv2, print_val_loss_XNetv2, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_2d, draw_pred_sup, print_best_sup
from config.visdom_config.visual_visdom import visdom_initialization_XNetv2, visualization_XNetv2, visual_image_sup
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_XNetv2
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ISIC-2017', help='CREMI, ISIC-2017, GlaS')
    parser.add_argument('--sup_mark', default='100', help='100')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.8, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=5, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--wavelet_type', default='haar', help='haar, db2, bior1.5, bior2.4, coif1, dmey')
    parser.add_argument('--train_alpha', default=[0.0, 0.4])
    parser.add_argument('--train_beta', default=[0.0, 0.4])
    parser.add_argument('--val_alpha', default=[0.2, 0.2])
    parser.add_argument('--val_beta', default=[0.2, 0.2])

    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='XNetv2', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672, help='16672')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    path_trained_models = cfg['PATH_TRAINED_MODEL'] + '/' + str(dataset_name)
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models + '/' + str(args.network) + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark)+str(args.train_alpha[0])+'-'+str(args.train_alpha[1])+'-'+str(args.train_beta[0])+'-'+str(args.train_beta[1])
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)

    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results + '/' + str(args.network) + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark)+str(args.train_alpha[0])+'-'+str(args.train_alpha[1])+'-'+str(args.train_beta[0])+'-'+str(args.train_beta[1])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)

    if args.vis and rank == args.rank_index:
        visdom_env = str('Sup-XNetv2-' + str(dataset_name) + '-' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size)+ '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark)+str(args.train_alpha[0])+'-'+str(args.train_alpha[1])+'-'+str(args.train_beta[0])+'-'+str(args.train_beta[1]))
        visdom = visdom_initialization_XNetv2(env=visdom_env, port=args.visdom_port)

    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_train = imagefloder_XNetv2(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_' + args.sup_mark,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.train_alpha,
        beta=args.train_beta,
        sup=True,
        num_images=None,
    )
    dataset_val = imagefloder_XNetv2(
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.val_alpha,
        beta=args.val_beta,
        sup=True,
        num_images=None,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=val_sampler)

    num_batches = {'train_sup': len(dataloaders['train']), 'val': len(dataloaders['val'])}

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    dist.barrier()

    # Training Strategy
    criterion = segmentation_loss(args.loss, False).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)

    # Train & Val
    since = time.time()
    count_iter = 0

    best_model = model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        dataloaders['train'].sampler.set_epoch(epoch)
        model.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_sup_3 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0
        val_loss_sup_3 = 0.0
        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
        dist.barrier()

        for i, data in enumerate(dataloaders['train']):

            inputs_train_1 = Variable(data['image'].cuda())
            inputs_train_2 = Variable(data['L'].cuda())
            inputs_train_3 = Variable(data['H'].cuda())
            mask_train = Variable(data['mask'].cuda())

            optimizer.zero_grad()
            outputs_train1, outputs_train2, outputs_train3 = model(inputs_train_1, inputs_train_2, inputs_train_3)
            torch.cuda.empty_cache()

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = outputs_train1
                    mask_list_train = mask_train
                # else:
                elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train1 = torch.cat((score_list_train1, outputs_train1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train), dim=0)

            max_train1 = torch.max(outputs_train1, dim=1)[1].long()
            max_train2 = torch.max(outputs_train2, dim=1)[1].long()
            max_train3 = torch.max(outputs_train3, dim=1)[1].long()

            loss_train_sup1 = criterion(outputs_train1, mask_train)
            loss_train_sup2 = criterion(outputs_train2, mask_train)
            loss_train_sup3 = criterion(outputs_train3, mask_train)

            loss_train_unsup = criterion(outputs_train1, max_train2) + criterion(outputs_train2, max_train1) + \
                               criterion(outputs_train1, max_train3) + criterion(outputs_train3, max_train1)
            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train = loss_train_sup1 + loss_train_sup2 + loss_train_sup3 + loss_train_unsup

            loss_train.backward()
            optimizer.step()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_sup_3 += loss_train_sup3.item()
            train_loss_unsup += loss_train_unsup.item()
            train_loss += loss_train.item()

        scheduler_warmup.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            score_gather_list_train1 = [torch.zeros_like(score_list_train1) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train1, score_list_train1)
            score_list_train1 = torch.cat(score_gather_list_train1, dim=0)

            mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            if rank == args.rank_index:
                torch.cuda.empty_cache()
                print('=' * print_num)
                print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
                train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_epoch_loss = print_train_loss_XNetv2(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
                train_eval_list1, train_m_jc1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
                torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()

                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val']:
                    inputs_val1 = Variable(data['image'].cuda())
                    inputs_val2 = Variable(data['L'].cuda())
                    inputs_val3 = Variable(data['H'].cuda())
                    mask_val = Variable(data['mask'].cuda())
                    name_val = data['ID']

                    optimizer.zero_grad()
                    outputs_val1, outputs_val2, outputs_val3 = model(inputs_val1, inputs_val2, inputs_val3)
                    torch.cuda.empty_cache()

                    if i == 0:
                        score_list_val1 = outputs_val1
                        mask_list_val = mask_val
                        name_list_val = name_val
                    else:
                        score_list_val1 = torch.cat((score_list_val1, outputs_val1), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)

                    loss_val_sup1 = criterion(outputs_val1, mask_val)
                    loss_val_sup2 = criterion(outputs_val2, mask_val)
                    loss_val_sup3 = criterion(outputs_val3, mask_val)
                    val_loss_sup_1 += loss_val_sup1.item()
                    val_loss_sup_2 += loss_val_sup2.item()
                    val_loss_sup_3 += loss_val_sup3.item()

                torch.cuda.empty_cache()
                score_gather_list_val1 = [torch.zeros_like(score_list_val1) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val1, score_list_val1)
                score_list_val1 = torch.cat(score_gather_list_val1, dim=0)

                mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                mask_list_val = torch.cat(mask_gather_list_val, dim=0)

                name_gather_list_val = [None for _ in range(ngpus_per_node)]
                torch.distributed.all_gather_object(name_gather_list_val, name_list_val)
                name_list_val = np.concatenate(name_gather_list_val, axis=0)
                torch.cuda.empty_cache()

                if rank == args.rank_index:
                    val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3 = print_val_loss_XNetv2(val_loss_sup_1, val_loss_sup_2, val_loss_sup_3, num_batches, print_num, print_num_minus)
                    val_eval_list1, val_m_jc1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val1, mask_list_val, print_num_minus)
                    best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, model, score_list_val1, name_list_val, val_eval_list1, path_trained_models, path_seg_results, cfg['PALETTE'], 'XNetv2')
                    torch.cuda.empty_cache()

                    if args.vis:
                        draw_img = draw_pred_sup(cfg['NUM_CLASSES'], mask_train, mask_val, outputs_train1, outputs_val1, train_eval_list1, val_eval_list1)
                        visualization_XNetv2(visdom, epoch+1, train_epoch_loss, train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_m_jc1, val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3, val_m_jc1)
                        visual_image_sup(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3])

                    print('-' * print_num)
                    print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
        torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)

        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
        print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
        print('=' * print_num)