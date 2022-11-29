import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import pytorch_warmup as warmup

sys.path.append("../")
from utils.utils import *
from utils import KD_loss
from utils.imagenet_data_dali import imagenet_loader_dali
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler.cosine_lr import CosineLRScheduler

import timm.models as models
import torch_optimizer as optim

from torchvision import datasets, transforms
from torch.autograd import Variable
from bnext import BNext, BasicBlock
from birealnet import BNext18

parser = argparse.ArgumentParser("bnext")
parser.add_argument('--model', type=str, default="bnext_tiny",
                    help="student model")
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=128,
                    help='num of training epochs')
parser.add_argument('--optimizer', type=str, default="AdamW",
                    help="the optimizer during training")
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--save', type=str, default='./models',
                    help='path for saving trained models')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1,
                    help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnext101_32x8d',
                    help='path of ImageNet')
parser.add_argument('--teacher_num', type=int, default=1,
                    help='number of teachers')
parser.add_argument('--assistant_teacher_num', type=int, default=1,
                    help='number of teachers')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset', type=str, default = "ImageNet",
                    help='experiment dataset')
parser.add_argument('--dali', action='store_true',
                    help='use dali as data decoder and argumentation')
parser.add_argument('--dali_cpu', action='store_true',
                    help="dali mode (cpu, gpu)")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--weak_teacher', default="ResNet101", type=str,
                    help="Define the weak teacher in multi teacher KD")
parser.add_argument('--multi_teachers', default=True, type=bool,
                    help="Use mixup to argument training dataset")
parser.add_argument('--distillation', default=False, type=bool,
                    help="Use mixup to argument training dataset")
parser.add_argument('--hard_knowledge', default=False, type=bool,
                    help="activate hard knowledge aware distillation")
parser.add_argument('--hard_knowledge_grains', default="Instance", type=str,
                    help="define the grains of hard aware knowledge distillation (Instance, Batch)")

# Dataset parameters
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0., metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', default='rand-m7-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=int, default=1, metavar="AUG-REPEATS",
                    help='Number of augmentation repetitions (distributed training only) (default: 3)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default= 0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.0,
                    help='Label smoothing (default: 0.0)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

#Model parameters
parser.add_argument('--elm-attention', default = True, type = bool,
                    help='enabling the elm-attention')
parser.add_argument('--infor-recoupling', default = True, type = bool,
                    help='enabling the infor-recoupling')


def adjust_temperature(model, epoch, args):
  temperature = torch.ones(1)
  if args.model == "bnext18":
      from birealnet import HardSign, HardBinaryConv
  else:
      from bnext import HardSign, HardBinaryConv

  for module in model.modules():
    if isinstance(module, (HardSign, HardBinaryConv)):
      if (epoch % 1)==0 and (epoch != 0):
          module.temperature.mul_(0.9)
      temperature = module.temperature
  return temperature
  
def otsu_loss(network):
    otsu_loss = []
    r_positive = [] 
    entropy = []
    for module in network.modules():
        if isinstance(module, Block):
          if module.otsu_loss != 0:
            otsu_loss.append(module.otsu_loss)
          if module.r_positive != 0:  
            r_positive.append(module.r_positive)
          if module.entropy != 0:  
            entropy.append(module.entropy)
    
    if len(otsu_loss) != 0:
        return sum(otsu_loss)/len(otsu_loss), sum(r_positive)/len(r_positive), sum(entropy)/len(entropy)
    
    else:
        return torch.zeros(1), torch.zeros(1), torch.zeros(1)

def adjust_sparse_rate(network, decay_ratio, epoch):
    sparse_rate = min(5, 1 + 4*(1 - (decay_ratio)**epoch))
    for module in network.modules():
        if isinstance(module, MultiHead_Embedding):
            module.alpha = torch.Tensor([sparse_rate]).cuda()

    return sparse_rate

def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def main_worker(gpu, ngpus_per_node, args):
    if args.dataset == "ImageNet":
        CLASSES = 1000
    elif args.dataset == "CIAR100":
        CLASSES = 100
    else:
        CLASSES = 10
    
    if args.aa != "rand-m7-mstd0.5-inc1":
        args.aa = None
    else:
        pass

    args.num_classes = CLASSES
    args.gpu = gpu
    args.local_rank = gpu
    print(args.distillation)

    if not torch.cuda.is_available():
        sys.exit(1)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        #print("tcp://127.0.0.1:{}".format(find_free_port()))
        #dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        print("### Finished group initialization")

    start_t = time.time()

    logging.info("args = %s", args)

    # load model
    if args.teacher_num >0:
        model_teacher = nn.ModuleList()
    
        assistant_teachers = nn.Sequential()
        if args.assistant_teacher_num >= 1:
            assistant_teachers.append(models.efficientnet_b0(pretrained=True))
            print("using efficient_b0 as assistant teacher") 

        if args.assistant_teacher_num >= 2:
            if args.model == "bnext_super":
                assistant_teachers.append(models.efficientnet_b4(pretrained=True))
                print("using efficient_b4 as assistant teacher")    
            else:
                assistant_teachers.append(models.efficientnet_b2(pretrained=True))
                print("using efficient_b2 as assistant teacher")
        
        if args.assistant_teacher_num >= 3:
            if args.model == "bnext_super":
                assistant_teachers.append(models.convnext_tiny(pretrained=True))
                print("using convnext_tiny as assistant teacher")
            else:
                assistant_teachers.append(models.efficientnet_b4(pretrained=True))
                print("using efficient_b4 as assistant teacher")

        if args.assistant_teacher_num >= 4:
            assistant_teacher.append(models.convnext_tiny(pretrained=True))
            print("Using convnext_tiny as assistant_teacher")

        if args.assistant_teacher_num > 0:
            model_teacher.append(assistant_teachers)
            print("loading assistant teachers")
        else:
            pass
        
        if args.assistant_teacher_num >= 3:
            model_teacher.append(models.convnext_tiny(pretrained=True))
            print("using convnext_tiny as main teacher")
        
        elif args.assistant_teacher_num == 2:
            model_teacher.append(models.efficientnet_b4(pretrained=True))
            print("using efficient_b4 as main teacher")
 
        elif args.assistant_teacher_num == 1:
            model_teacher.append(models.efficientnet_b2(pretrained=True))
            print("using efficient_b2 as main teacher")
        
        else:
            if args.model == "bnext_large" or args.model == "bnext_super":
                model_teacher.append(models.convnext_tiny(pretrained = True))
                print("using convnext_tiny as main teacher")
            else:
                model_teacher.append(models.efficientnet_b0(pretrained=True))
                print("using efficient_b0 as main teacher")
            
        for teacher in model_teacher:
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
    else:
        model_teacher = None
    
    if model_teacher is not None and len(model_teacher)==2:
        args.hard_knowledge=True
    
    else:
        args.hard_knowledge=False

    if args.model == "bnext_tiny":
        model_student = BNext(num_classes = CLASSES, size = "tiny", ELM_Attention = args.elm_attention, Infor_Recoupling = args.infor_recoupling)
    elif args.model == "bnext_small":
        model_student = BNext(num_classes = CLASSES, size = "small", ELM_Attention = args.elm_attention, Infor_Recoupling = args.infor_recoupling)
    elif args.model == "bnext_middle":
        model_student = BNext(num_classes = CLASSES, size = "middle", ELM_Attention = args.elm_attention, Infor_Recoupling = args.infor_recoupling)
    elif args.model == "bnext_large":
        model_student = BNext(num_classes = CLASSES, size = "large", ELM_Attention = args.elm_attention, Infor_Recoupling = args.infor_recoupling)
    elif args.model == "bnext_super":
        model_student = BNext(num_classes = CLASSES, size = "super", ELM_Attention = args.elm_attention, Infor_Recoupling = args.infor_recoupling)
    elif args.model == "bnext18":
        model_student = BNext18(num_classes = CLASSES)
    else:
        raise ValueError("network not defined")

    if args.local_rank == 0:
        print(model_student)
    
    logging.info('student:')
    logging.info(model_student)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            if model_teacher is not None:
                for teacher in model_teacher:
                    teacher.cuda(args.gpu)

            model_student.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1)/ ngpus_per_node)
            if args.sync_bn:
                model_student = nn.SyncBatchNorm.convert_sync_batchnorm(model_student)
            model_student = torch.nn.parallel.DistributedDataParallel(model_student, device_ids=[args.gpu], find_unused_parameters=True)
            #model_teacher = model_teacher
        else:
            model_student.cuda()
            if args.teacher is not None:
                for teacher in model_teacher:
                    teacher.cuda()

            model_teacher = model_teacher
            if args.sync_bn:
                model_student = nn.SyncBatchNorm.convert_sync_batchnorm(model_student)
            model_student = torch.nn.parallel.DistributedDataParallel(model_student, find_unused_parameters=True)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_student = model_student.cuda(args.gpu)
        if model_teacher is not None:
            for teacher in model_teacher:
                teacher = teacher.cuda(args.gpu)
    else:
        print("Using DataParallel")    
        model_student = torch.nn.DataParallel(model_student).cuda()
        if model_teacher is not None:
            for teacher in model_teacher:
                teacher = torch.nn.DataParallel(teacher).cuda()

    #model_student = nn.DataParallel(model_student).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = criterion.cuda(args.gpu)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth).cuda(args.gpu)
    criterion_smooth = criterion_smooth.cuda(args.gpu)
    criterion_kd = KD_loss.DistributionLoss().cuda(args.gpu)

    all_parameters = model_student.parameters()
    weight_parameters = []
    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 or 'binary_conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    attention_parameters = []
    for pname, p in model_student.named_parameters():
        if 'binary_conv' in pname:
            attention_parameters.append(p)
    
    normal_parameters = []
    normal_parameters = list(filter(lambda p: id(p) not in (weight_parameters_id + weight_parameters_id), all_parameters))

    
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            [{'params' : other_parameters, 'weight_decay': 1e-3},
             {'params' : weight_parameters, 'weight_decay' : 1e-8}],
            lr=args.learning_rate,)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
                [{'params' : other_parameters, 'weight_decay': 1e-5},
                 {'params' : weight_parameters, 'weight_decay': 1e-8}],
                lr=args.learning_rate)
    else:
        raise ValueError

    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, warmup_t=5, lr_min = 1e-7, warmup_lr_init=1e-5, warmup_prefix = True)
    
    start_epoch = 0
    best_top1_acc= 0

    training_loss = []
    training_top1 = []
    training_top5 = []

    training_temperature = []

    testing_loss = []
    testing_top1 = []
    testing_top5 = []
  
    checkpoint_tar = os.path.join(args.save + "_{}_optimizer_{}_mixup_{}_cutmix_{}_aug_repeats_{}_KD_{}_assistant_{}_{}_HK_{}_{}_aa_{}__elm_{}_recoup_{}_0_amp".format(args.model, args.optimizer, args.mixup, args.cutmix, args.aug_repeats, args.teacher_num, args.assistant_teacher_num, args.weak_teacher, args.hard_knowledge, args.hard_knowledge_grains, args.aa, args.elm_attention, args.infor_recoupling), 'checkpoint.pth.tar')
    
    print(checkpoint_tar)
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        if args.gpu is None:
            checkpoint = torch.load(checkpoint_tari, map_location="cpu")
        else:
            # Map model to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(checkpoint_tar, map_location=loc)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_top1_acc = best_top1_acc
        model_student.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

        training_loss = checkpoint["train_loss"]
        training_top1 = checkpoint["train_top1"]
        training_top5 = checkpoint["train_top5"]

        testing_loss = checkpoint["test_loss"]
        testing_top1 = checkpoint["test_top1"]
        testing_top5 = checkpoint["test_top5"]
        print("done")
    else:
        print("checkpoint not exists")

    cudnn.benchmark = True

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step(epoch = epoch)

    if args.dataset == "ImageNet":
        
        print("Training on ImageNet")
        
        if args.dali:
            train_loader, val_loader = imagenet_loader_dali(args)
            
        # load training data
        else:
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD

            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                         std=[0.229, 0.224, 0.225])
            normalize = transforms.Normalize(mean = mean, std = std)
        # data augmentation
            crop_scale = 0.08
            lighting_param = 0.1
            
            dataset_train = create_dataset(
                args.dataset, root=traindir, split=args.train_split, is_training=True,
                class_map="",
                download=False,
                batch_size=args.batch_size,
                repeats=0)

            dataset_eval = create_dataset(
                args.dataset, root=valdir, split=args.val_split, is_training=False,
                class_map="",
                download=False,
                batch_size=args.batch_size)

            args.prefetcher = not args.no_prefetcher

            train_loader = create_loader(
                    dataset_train,
                    input_size = [3, 224, 224],
                    batch_size = args.batch_size,
                    is_training = True,
                    use_prefetcher=args.prefetcher,
                    no_aug=args.no_aug,
                    re_prob=args.reprob,
                    re_mode=args.remode,
                    re_count=args.recount,
                    re_split=args.resplit,
                    scale=args.scale,
                    ratio=args.ratio,
                    hflip=args.hflip,
                    vflip=args.vflip,
                    color_jitter=args.color_jitter,
                    auto_augment=args.aa,
                    num_aug_repeats=args.aug_repeats,
                    num_aug_splits=args.aug_splits,
                    interpolation=args.train_interpolation,
                    mean=mean,
                    std=std,
                    num_workers=args.workers,
                    distributed=args.distributed,
                    collate_fn=None,
                    pin_memory=args.pin_mem,
                    use_multi_epochs_loader=args.use_multi_epochs_loader,
                    worker_seeding=args.worker_seeding,
                    )
            
            # load validation data
            
            val_loader = create_loader(
                dataset_eval,
                input_size=[3, 224, 224],
                batch_size=args.batch_size*2,
                is_training=False,
                use_prefetcher=args.prefetcher,
                interpolation=args.train_interpolation,
                mean=mean,
                std=std,
                num_workers=args.workers,
                distributed=False,
                crop_pct=0.95,
                pin_memory=args.pin_mem,
            )
            
    else:
        print("Training on CIFAR")
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std = [0.267, 0.256, 0.276])
        
        #train data arguments
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        
        #load train datasets
        if args.dataset == "CIFAR100":
            train_dataset = datasets.CIFAR100(root = "datasets/CIFAR100/", train = True, download = True, transform = train_transforms)
        else:
            train_dataset = datasets.CIFAR10(root = "datasets/CIFAR10/", train = True, download = True, transform = train_transforms)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        
        #val data arguments
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        #load validation data
        if args.dataset == "CIFAR100":
            val_dataset = datasets.CIFAR100(root = "datasets/CIFAR100/", train = False, download = True, transform = val_transforms)
        else:
            val_dataset = datasets.CIFAR10(root = "datasets/CIFAR10/", train = False, download = True, transform = val_transforms)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # train the model
    epoch = start_epoch

    temperature = 1.0
    training_temperature.append(temperature)

    #if not args.multiprocessing_distributed or args.local_rank == 0:

    valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model_student, criterion, args)


    while epoch < args.epochs:
        if args.dataset == "ImageNet" and args.teacher_num > 0:
            train_obj, train_top1_acc,  train_top5_acc, alpha = train(epoch,  train_loader, model_student, model_teacher, criterion_kd, optimizer, scheduler, temperature, args)
        else:
            train_obj, train_top1_acc,  train_top5_acc, alpha = train(epoch,  train_loader, model_student, None, criterion, optimizer, scheduler, temperature, args)
        
        if not args.multiprocessing_distributed or args.local_rank == 0:
            valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model_student, criterion, args)

        if not args.multiprocessing_distributed or args.local_rank == 0:
            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True
       
            print("Best Acc:{}%, Temp: {}".format(best_top1_acc, temperature))
        
        if not args.multiprocessing_distributed or args.local_rank == 0:
            training_loss.append(train_obj)
            training_top1.append(train_top1_acc)
            training_top5.append(train_top5_acc)

            testing_loss.append(valid_obj)
            testing_top1.append(valid_top1_acc)
            testing_top5.append(valid_top5_acc)
        
        if not args.multiprocessing_distributed or args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch,
                'train_loss': training_loss,
                'train_top1': training_top1, 
                'train_top5': training_top5,
                'test_loss': testing_loss,
                'test_top1': testing_top1,
                'test_top5': testing_top5,
                'state_dict': model_student.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer' : optimizer.state_dict(),
                'temp': training_temperature,
                'alpha': alpha,
                }, is_best, args.save + "_" + "{}_optimizer_{}_mixup_{}_cutmix_{}_aug_repeats_{}_KD_{}_assistant_{}_{}_HK_{}_{}_aa_{}__elm_{}_recoup_{}_{}_amp".format(args.model, args.optimizer, args.mixup, args.cutmix, args.aug_repeats, args.teacher_num, args.assistant_teacher_num, args.weak_teacher, args.hard_knowledge, args.hard_knowledge_grains, args.aa, args.elm_attention, args.infor_recoupling, args.gpu, args.epochs), epoch = epoch)

        epoch += 1
        
        temperature = adjust_temperature(model_student, epoch, args).item()
        training_temperature.append(temperature)

        if args.dali and args.dataset == "ImageNet":
            train_loader.reset()
            val_loader.reset()
        
    training_time = (time.time() - start_t) / 3600
    if not args.multiprocessing_distributed or args.local_rank == 0:
        print('total training time = {} hours'.format(training_time))
        print("best acc: {}".format(best_top1_acc))


def train(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, temperature, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    losses_all = AverageMeter('Loss_ALL', ':.4e')
    losses_entropy = AverageMeter('Loss_Entropy', ':.4e')

    ratio_positive = AverageMeter('Ratio_positive', ':.4e')
    binary_entropy = AverageMeter('entropy', ':.3e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    alpha_beta = AverageMeter('Alpha_Beta', ':6.2f')

    sparse_ratio = 1.0
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_all, binary_entropy, alpha_beta, top1, top5],
        prefix="Epoch: [{}] PR: [{:.2e}] ".format(epoch, sparse_ratio))

    model_student.train()
    if model_teacher is not None:
        for teacher in model_teacher:
            teacher.eval()

    end = time.time()
    scheduler.step(epoch = epoch)
    optimizer.zero_grad()

    scaler = torch.cuda.amp.GradScaler()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    if args.mixup > 0 or args.cutmix > 0:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        
        mixup = Mixup(**mixup_args)
    else:
        mixup = None

    if len(model_teacher) >=2:
        if epoch /(args.epochs/args.assistant_teacher_num) < 1 and args.assistant_teacher_num > 0:
            model_teacher_r = [model_teacher[0][0]] + [model_teacher[1]]
        elif epoch /(args.epochs/args.assistant_teacher_num) < 2 and args.assistant_teacher_num > 1:
            model_teacher_r = [model_teacher[0][1]] + [model_teacher[1]]
        elif  epoch /(args.epochs/args.assistant_teacher_num) < 3 and args.assistant_teacher_num > 2:
            model_teacher_r = [model_teacher[0][2]] + [model_teacher[1]]
        else:
             model_teacher_r = [model_teacher[0][-1]] + [model_teacher[1]]
    else:
        model_teacher_r = model_teacher
        

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.dataset == "ImageNet" and args.dali:
            images = batch[0]['data'].cuda()
            target = batch[0]['label'].squeeze(-1).long().cuda()
        
        else:
            images = batch[0].cuda()
            target = batch[1].cuda()
       
        if mixup is not None:
            images, target = mixup(images, target)
       
        logits_student = model_student(images)
    
        alpha = []
        
        if model_teacher is not None:
            loss_entropy = []
            logits_teachers = []
            
            for teacher in model_teacher_r:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        logits_teacher = (teacher(images).detach())
                logits_teacher=logits_teacher.float()
                logits_teachers.append(logits_teacher)
                loss_entropy.append((criterion(logits_student, logits_teacher, reduce = (False if args.hard_knowledge_grains == "instance" else True))))
                
           
            for item in logits_teachers:
                alpha.append(nn.CrossEntropyLoss(reduce = (False if args.hard_knowledge_grains == "Instance" else True)).cuda()(logits_student, torch.softmax(item, dim = -1)).unsqueeze(-1))
           
            alpha = torch.softmax(torch.cat(alpha, dim = -1), dim = -1).detach()
            
            if args.hard_knowledge is False:
                alpha = torch.ones_like(alpha)*0.5
                    
            beta = (1 - alpha).detach()
            
            loss = []
            for i in range(len(logits_teachers)):
                loss.append(loss_entropy[i]*alpha[:,i])

            loss = torch.mean(sum(loss)/len(loss))
    
        else:
            loss = criterion(logits_student, target)
            
            alpha = torch.zeros(1)

        loss_all = loss
        
        # measure accuracy and record loss
        if mixup is None:
            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        else:
            prec1, prec5 = torch.zeros(1), torch.zeros(1)
        
        n = images.size(0)
        
        losses_all.update(loss_all.item(), n)   #accumulated loss
        losses_entropy.update(loss.item(), n)

        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        
        if args.hard_knowledge is not False:
            alpha_beta.update((alpha[:,0].view(-1).mean().item()), n)
        else:
            alpha_beta.update(alpha.mean(), n)

        # compute gradient and do SGD step 
        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(model_student.parameters(), 5)
        
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    return losses_all.avg, top1.avg, top5.avg, alpha_beta.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            if args.dataset == "ImageNet" and args.dali:
                images = batch[0]["data"]
                target = batch[0]["label"].squeeze(-1).long().cuda()
            else:
                images = batch[0].cuda()
                target = batch[1].cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0].item(), n)
            top5.update(pred5[0].item(), n)

            torch.cuda.synchronize()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            progress.display(i)

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def main():
    args = parser.parse_args()

    args.save = "." + "models/ImageNet" if args.dataset == "ImageNet" else "models/CIFAR100"
    
    if not os.path.exists('log'):
        os.mkdir('log')

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join('log/log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    print("#### {} GPUs per Node".format(ngpus_per_node))

    if args.multiprocessing_distributed:
        print("##### Multiprocessing_distributed Training")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("##### Training")
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

if __name__ == '__main__':
    main()
