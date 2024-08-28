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
import torchio as tio
import pywt
from torchio import transforms as T

from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import print_train_loss_XNetv2, print_val_loss_XNetv2, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_3d, print_best_sup
from config.visdom_config.visual_visdom import visdom_initialization_XNetv2, visualization_XNetv2
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_3d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_it
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
    parser.add_argument('--dataset_name', default='P-CT', help='P-CT, LiTS')
    parser.add_argument('--sup_mark', default='20')
    parser.add_argument('--unsup_mark', default='80')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-e', '--num_epochs', default=300, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.05, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=3, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--wavelet_type', default='bior1.5', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--train_alpha', default=[0, 0.4])
    parser.add_argument('--train_beta', default=[0.5, 0.8])
    parser.add_argument('--val_alpha', default=[0.2, 0.2])
    parser.add_argument('--val_beta', default=[0.65, 0.65])

    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('-n', '--network', default='XNetv2_3D_min', type=str)
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

    path_trained_models = cfg['PATH_TRAINED_MODEL']+'/'+str(dataset_name)
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models+'/'+args.network+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s=' + str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-uw='+str(args.unsup_weight)+'-w=' + str(args.warm_up_duration)+'-'+str(args.sup_mark)+'-'+str(args.unsup_mark)
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)

    if args.vis and rank == args.rank_index:
        visdom_env = str('Semi-XNetv2-' + str(dataset_name) + '-' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size)+'-uw='+str(args.unsup_weight) + '-w=' + str(args.warm_up_duration)+'-'+str(args.sup_mark)+'-'+str(args.unsup_mark))
        visdom = visdom_initialization_XNetv2(env=visdom_env, port=args.visdom_port)

    # Dataset
    data_transform = data_transform_3d(cfg['NORMALIZE'])

    dataset_train_unsup = dataset_it(
        data_dir=cfg['PATH_DATASET'] + '/train_unsup_' + args.unsup_mark,
        transform_1=data_transform['train'],
        queue_length=cfg['QUEUE_LENGTH'],
        samples_per_volume=cfg['NUM_SAMPLE_TRAIN'],
        patch_size=cfg['PATCH_SIZE'],
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=False,
        num_images=None
    )
    num_images_unsup = len(dataset_train_unsup.dataset_1)

    dataset_train_sup = dataset_it(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_' + args.sup_mark,
        transform_1=data_transform['train'],
        queue_length=cfg['QUEUE_LENGTH'],
        samples_per_volume=cfg['NUM_SAMPLE_TRAIN'],
        patch_size=cfg['PATCH_SIZE'],
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=True,
        num_images=num_images_unsup
    )
    dataset_val = dataset_it(
        data_dir=cfg['PATH_DATASET'] + '/val',
        transform_1=data_transform['val'],
        queue_length=cfg['QUEUE_LENGTH'],
        samples_per_volume=cfg['NUM_SAMPLE_VAL'],
        patch_size=cfg['PATCH_SIZE'],
        num_workers=8,
        shuffle_subjects=False,
        shuffle_patches=False,
        sup=True,
        num_images=None
    )

    train_sampler_unsup = torch.utils.data.distributed.DistributedSampler(dataset_train_unsup.queue_train_set_1, shuffle=True)
    train_sampler_sup = torch.utils.data.distributed.DistributedSampler(dataset_train_sup.queue_train_set_1, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val.queue_train_set_1, shuffle=False)

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=train_sampler_sup)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=train_sampler_unsup)
    dataloaders['val'] = DataLoader(dataset_val.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=val_sampler)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    # Model
    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])

    model1 = model1.cuda()
    model1 = DistributedDataParallel(model1, device_ids=[args.local_rank])
    dist.barrier()

    # Training Strategy
    criterion = segmentation_loss(args.loss, False).cuda()

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    # L H augmentation
    L_H_aug = T.Compose([T.Resize(cfg['PATCH_SIZE']), T.ZNormalization(masking_method=cfg['NORMALIZE'])])

    # Train & Val
    since = time.time()
    count_iter = 0

    best_model = model1
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        dataloaders['train_sup'].sampler.set_epoch(epoch)
        dataloaders['train_unsup'].sampler.set_epoch(epoch)
        model1.train()

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

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup_1 = Variable(unsup_index['image'][tio.DATA].cuda())

            img_train_unsup_2 = torch.zeros_like(img_train_unsup_1, device='cpu')
            img_train_unsup_3 = torch.zeros_like(img_train_unsup_1, device='cpu')
            img_train_unsup_numpy = img_train_unsup_1.cpu().detach().numpy()
            for j in range(img_train_unsup_numpy.shape[0]):
                img = img_train_unsup_numpy[j, 0]
                img_wavelet = pywt.dwtn(img, args.wavelet_type, axes=(-3, -2, -1))
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = img_wavelet['aaa'], img_wavelet['aad'], img_wavelet['ada'], img_wavelet['add'], img_wavelet['daa'], img_wavelet['dad'], img_wavelet['dda'], img_wavelet['ddd']

                LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) * 255
                LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min()) * 255
                LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min()) * 255
                LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min()) * 255
                HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min()) * 255
                HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min()) * 255
                HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min()) * 255
                HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min()) * 255

                H_ = LLH + LHL + LHH + HLL + HLH + HHL + HHH
                H_ = (H_ - H_.min()) / (H_.max() - H_.min()) * 255

                L_alpha = random.uniform(args.train_alpha[0], args.train_alpha[1])
                L = LLL + L_alpha * H_
                L = (L - L.min()) / (L.max() - L.min()) * 255

                H_beta = random.uniform(args.train_beta[0], args.train_beta[1])
                H = H_ + H_beta * LLL
                H = (H - H.min()) / (H.max() - H.min()) * 255

                L = torch.tensor(L).unsqueeze(0)
                H = torch.tensor(H).unsqueeze(0)
                L = L_H_aug(L)
                H = L_H_aug(H)
                img_train_unsup_2[j] = L
                img_train_unsup_3[j] = H
            img_train_unsup_2 = Variable(img_train_unsup_2.cuda())
            img_train_unsup_3 = Variable(img_train_unsup_3.cuda())
            torch.cuda.empty_cache()

            optimizer1.zero_grad()
            pred_train_unsup1, pred_train_unsup2, pred_train_unsup3 = model1(img_train_unsup_1, img_train_unsup_2, img_train_unsup_3)

            max_train_unsup1 = torch.max(pred_train_unsup1, dim=1)[1].long()
            max_train_unsup2 = torch.max(pred_train_unsup2, dim=1)[1].long()
            max_train_unsup3 = torch.max(pred_train_unsup3, dim=1)[1].long()

            loss_train_unsup = criterion(pred_train_unsup1, max_train_unsup2) + \
                               criterion(pred_train_unsup2, max_train_unsup1) + \
                               criterion(pred_train_unsup1, max_train_unsup3) + \
                               criterion(pred_train_unsup3, max_train_unsup1)
            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)
            torch.cuda.empty_cache()

            sup_index = next(dataset_train_sup)
            img_train_sup_1 = Variable(sup_index['image'][tio.DATA].cuda())

            img_train_sup_2 = torch.zeros_like(img_train_sup_1, device='cpu')
            img_train_sup_3 = torch.zeros_like(img_train_sup_1, device='cpu')
            img_train_sup_numpy = img_train_sup_1.cpu().detach().numpy()
            for j in range(img_train_sup_numpy.shape[0]):
                img = img_train_sup_numpy[j, 0]
                img_wavelet = pywt.dwtn(img, args.wavelet_type)
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = img_wavelet['aaa'], img_wavelet['aad'], img_wavelet['ada'], img_wavelet['add'], img_wavelet['daa'], img_wavelet['dad'], img_wavelet['dda'], img_wavelet['ddd']

                LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) * 255
                LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min()) * 255
                LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min()) * 255
                LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min()) * 255
                HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min()) * 255
                HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min()) * 255
                HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min()) * 255
                HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min()) * 255

                H_ = LLH + LHL + LHH + HLL + HLH + HHL + HHH
                H_ = (H_ - H_.min()) / (H_.max() - H_.min()) * 255

                L_alpha = random.uniform(args.train_alpha[0], args.train_alpha[1])
                L = LLL + L_alpha * H_
                L = (L - L.min()) / (L.max() - L.min()) * 255

                H_beta = random.uniform(args.train_beta[0], args.train_beta[1])
                H = H_ + H_beta * LLL
                H = (H - H.min()) / (H.max() - H.min()) * 255

                L = torch.tensor(L).unsqueeze(0)
                H = torch.tensor(H).unsqueeze(0)
                L = L_H_aug(L)
                H = L_H_aug(H)
                img_train_sup_2[j] = L
                img_train_sup_3[j] = H
            img_train_sup_2 = Variable(img_train_sup_2.cuda())
            img_train_sup_3 = Variable(img_train_sup_3.cuda())
            mask_train_sup = Variable(sup_index['mask'][tio.DATA].squeeze(1).long().cuda())
            torch.cuda.empty_cache()

            pred_train_sup1, pred_train_sup2, pred_train_sup3 = model1(img_train_sup_1, img_train_sup_2, img_train_sup_3)

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    mask_list_train = mask_train_sup
                # else:
                elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = criterion(pred_train_sup1, mask_train_sup)
            loss_train_sup2 = criterion(pred_train_sup2, mask_train_sup)
            loss_train_sup3 = criterion(pred_train_sup3, mask_train_sup)
            loss_train_sup = loss_train_sup1 + loss_train_sup2 + loss_train_sup3
            loss_train_sup.backward()

            optimizer1.step()
            torch.cuda.empty_cache()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_sup_3 += loss_train_sup3.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()
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
                train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_sup_3, train_epoch_loss_unsup, train_epoch_loss = print_train_loss_XNetv2(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
                train_eval_list_1, train_m_jc_1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
                torch.cuda.empty_cache()

            with torch.no_grad():
                model1.eval()
                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val']:

                    inputs_val_1 = Variable(data['image'][tio.DATA].cuda().cuda())

                    inputs_val_2 = torch.zeros_like(inputs_val_1, device='cpu')
                    inputs_val_3 = torch.zeros_like(inputs_val_1, device='cpu')
                    inputs_val_numpy = inputs_val_1.cpu().detach().numpy()
                    for j in range(inputs_val_numpy.shape[0]):
                        img = inputs_val_numpy[j, 0]
                        img_wavelet = pywt.dwtn(img, args.wavelet_type)
                        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = img_wavelet['aaa'], img_wavelet['aad'], img_wavelet['ada'], img_wavelet['add'], img_wavelet['daa'], img_wavelet['dad'], img_wavelet['dda'], img_wavelet['ddd']

                        LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) * 255
                        LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min()) * 255
                        LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min()) * 255
                        LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min()) * 255
                        HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min()) * 255
                        HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min()) * 255
                        HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min()) * 255
                        HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min()) * 255

                        H_ = LLH + LHL + LHH + HLL + HLH + HHL + HHH
                        H_ = (H_ - H_.min()) / (H_.max() - H_.min()) * 255

                        L_alpha = random.uniform(args.val_alpha[0], args.val_alpha[1])
                        L = LLL + L_alpha * H_
                        L = (L - L.min()) / (L.max() - L.min()) * 255

                        H_beta = random.uniform(args.val_beta[0], args.val_beta[1])
                        H = H_ + H_beta * LLL
                        H = (H - H.min()) / (H.max() - H.min()) * 255

                        L = torch.tensor(L).unsqueeze(0)
                        H = torch.tensor(H).unsqueeze(0)
                        L = L_H_aug(L)
                        H = L_H_aug(H)
                        inputs_val_2[j] = L
                        inputs_val_3[j] = H
                    inputs_val_2 = Variable(inputs_val_2.cuda())
                    inputs_val_3 = Variable(inputs_val_3.cuda())
                    mask_val = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())
                    torch.cuda.empty_cache()

                    optimizer1.zero_grad()
                    outputs_val_1, outputs_val_2, outputs_val_3 = model1(inputs_val_1, inputs_val_2, inputs_val_3)
                    torch.cuda.empty_cache()

                    if i == 0:
                        score_list_val_1 = outputs_val_1
                        mask_list_val = mask_val
                    else:
                        score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)

                    loss_val_sup_1 = criterion(outputs_val_1, mask_val)
                    loss_val_sup_2 = criterion(outputs_val_2, mask_val)
                    loss_val_sup_3 = criterion(outputs_val_3, mask_val)
                    val_loss_sup_1 += loss_val_sup_1.item()
                    val_loss_sup_2 += loss_val_sup_2.item()
                    val_loss_sup_3 += loss_val_sup_3.item()

                torch.cuda.empty_cache()
                score_gather_list_val_1 = [torch.zeros_like(score_list_val_1) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val_1, score_list_val_1)
                score_list_val_1 = torch.cat(score_gather_list_val_1, dim=0)

                mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                mask_list_val = torch.cat(mask_gather_list_val, dim=0)
                torch.cuda.empty_cache()

                if rank == args.rank_index:
                    val_epoch_loss_sup_1, val_epoch_loss_sup_2, val_epoch_loss_sup_3 = print_val_loss_XNetv2(val_loss_sup_1, val_loss_sup_2, val_loss_sup_3, num_batches, print_num, print_num_minus)
                    val_eval_list_1, val_m_jc_1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val_1, mask_list_val, print_num_minus)
                    best_val_eval_list = save_val_best_sup_3d(best_val_eval_list, model1, val_eval_list_1, path_trained_models, 'XNetv2')
                    torch.cuda.empty_cache()

                    if args.vis:
                        visualization_XNetv2(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_sup_3, train_epoch_loss_unsup, train_m_jc_1, val_epoch_loss_sup_1, val_epoch_loss_sup_2, val_epoch_loss_sup_3, val_m_jc_1)
                    print('-' * print_num)
                    print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
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