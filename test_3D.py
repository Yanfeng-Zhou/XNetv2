from torchvision import transforms, datasets
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
from torch.backends import cudnn
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchio as tio
import pywt
from torchio import transforms as T

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_3d
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_it
from config.train_test_config.train_test_config import save_test_3d
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
    parser.add_argument('-p', '--path_model', default='.../XNetv2/checkpoints/P-CT/.../best_XNetv2_Jc_0.7329.pth')
    parser.add_argument('--dataset_name', default='P-CT', help='P-CT, LiTS')
    parser.add_argument('--threshold', default=0.2)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('--wavelet_type', default='bior1.5', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--val_alpha', default=[0.2, 0.2])
    parser.add_argument('--val_beta', default=[0.65, 0.65])
    parser.add_argument('-n', '--network', default='XNetv2')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    # Config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # Results Save
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results + '/' + str(os.path.splitext(os.path.split(args.path_model)[1])[0])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)

    data_transform = data_transform_3d(cfg['NORMALIZE'])
    dataset_val = dataset_it(
        data_dir=cfg['PATH_DATASET'] + '/val',
        transform_1=data_transform['test'],
    )

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()

    # if rank == args.rank_index:
    #     state_dict = torch.load(args.path_model, map_location=torch.device(args.local_rank))
    #     model.load_state_dict(state_dict=state_dict)
    # model = DistributedDataParallel(model, device_ids=[args.local_rank])

    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    state_dict = torch.load(args.path_model)
    model.load_state_dict(state_dict=state_dict)
    dist.barrier()

    # L H augmentation
    L_H_aug = T.Compose([T.Resize(cfg['PATCH_SIZE']), T.ZNormalization(masking_method=cfg['NORMALIZE'])])

    # Test
    since = time.time()

    for i, subject in enumerate(dataset_val.dataset_1):

        grid_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=cfg['PATCH_SIZE'],
            patch_overlap=cfg['PATCH_OVERLAP']
        )

        # val_sampler = torch.utils.data.distributed.DistributedSampler(grid_sampler, shuffle=False)

        dataloaders = dict()
        dataloaders['test'] = DataLoader(grid_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
        # dataloaders['test'] = DataLoader(grid_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16, sampler=val_sampler)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

        with torch.no_grad():
            model.eval()

            for data in dataloaders['test']:

                inputs_test1 = Variable(data['image'][tio.DATA].cuda())
                inputs_test2 = torch.zeros_like(inputs_test1, device='cpu')
                inputs_test3 = torch.zeros_like(inputs_test1, device='cpu')
                inputs_test_numpy = inputs_test1.cpu().detach().numpy()
                for j in range(inputs_test_numpy.shape[0]):
                    img = inputs_test_numpy[j, 0]
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
                    inputs_test2[j] = L
                    inputs_test3[j] = H
                inputs_test2 = Variable(inputs_test2.cuda())
                inputs_test3 = Variable(inputs_test3.cuda())

                location_test = data[tio.LOCATION]
                outputs_test_1, outputs_test_2, outputs_test_3 = model(inputs_test1, inputs_test2, inputs_test3)
                aggregator.add_batch(outputs_test_1, location_test)

        outputs_tensor = aggregator.get_output_tensor()
        save_test_3d(cfg['NUM_CLASSES'], outputs_tensor, subject['ID'], args.threshold, path_seg_results, subject['image']['affine'])


    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)