import argparse
import copy
import numpy as np
import os
import time
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import mlflow
import timm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from gradcam import GradCAM, process_val_images
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_finetune import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('Pytorch Image Classfication', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--hf_token', default=None, type=str)

    # Model parameters
    #parser.add_argument('--model_name', default='resnet18.a1_in1k', type=str, metavar='MODEL')
    parser.add_argument('--model_name', default='efficientnet_b1.ft_in1k', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=1536, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--ckpt_path', default=None, type=str)
    # Dataset parameters
   
    parser.add_argument('--data_path', required=True, type=str) # train / val 로 구분되어야함
    parser.add_argument('--nb_classes', default=5, type=int)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',type=str)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
    parser.add_argument('--pin_mem', action='store_true')


    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=1e5, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N')

    # MLflow parameters
    parser.add_argument('--mlflow_server' ,type=str, default = None, help='MLflow tracking server URI')
    parser.add_argument('--mlflow_experiment', type=str, default=None, help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None, help='MLflow run name')
    parser.add_argument('--mlflow_account_name', type=str, default=None, help='MLflow account ID')
    parser.add_argument('--mlflow_account_password', type=str, default=None , help='MLflow account password')
    parser.add_argument('--zip_file_name', type=str, default="fold1_gradcam_results", help='GradCAM file name')


    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--smoothing', type=float, default=0.1)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.0,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')



    parser.set_defaults(pin_mem=True)

    # Distributed training parameters  
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:12345', help='url used to set up distributed training')

    return parser


def main(args):
    os.environ["MLFLOW_TRACKING_USERNAME"] = args.mlflow_account_name
    os.environ["MLFLOW_TRACKING_PASSWORD"] = args.mlflow_account_password
    os.environ["HF_TOKEN"] = args.hf_token

    mlflow.set_tracking_uri(args.mlflow_server)
    mlflow.set_experiment(args.mlflow_experiment)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(vars(args))
        if args.distributed == False:
            pass
        else:
            misc.init_distributed_mode(args)

        device = torch.device(args.device)

        # Fix the seed for reproducibility
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        cudnn.benchmark = True

        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

        if args.distributed is not False:
            sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
            data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
            )
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val, batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
            )

        else:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, batch_size=args.batch_size,shuffle=True,
                num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
            )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, batch_size=args.batch_size,shuffle=False,
                num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
            )
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
        print(args.model_name)
        if args.pretrained is True:
            if args.ckpt_path is None:
                model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=args.nb_classes)
            else:
                model = torch.load(args.ckpt_path)
                if "efficientnet" in args.model_name.lower():
                    # For EfficientNet models, the classifier is named 'classifier'
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, args.nb_classes)
                elif "mobilenet" in args.model_name.lower():
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, args.nb_classes) 
                elif "resnet" in args.model_name.lower():
                    # For ResNet models, the classifier is named 'fc'
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, args.nb_classes)
                elif "deit" in args.model_name.lower() or "vit" in args.model_name.lower():
                    # For Vision Transformers and DeiT models
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, args.nb_classes)
                #model.fc = nn.Linear(2048, args.nb_classes, bias=True)
       
        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.blr,weight_decay=args.weight_decay)
        loss_scaler = NativeScaler()

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        best_acc=0
        best_f1=0
        best_epo=0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            _,train_loss = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,loss_scaler,args.clip_grad,mixup_fn, args=args
            )
            test_stats,precision,recall,f1 = evaluate(data_loader_val, model, device,args)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_acc_top1={test_stats['acc1']:.2f}%, f1={f1:.2f}%, precision={precision:.2f}%, recall={recall:.2f}%")
            mlflow.log_metrics({
                'train_loss': train_loss,
                'test_acc_top1': test_stats['acc1'],
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }, step=epoch)
            # if (epoch + 1) % 100 == 0:
            #     print(f"Saving model at epoch {epoch} (100 epoch checkpoint)...")
            #     mlflow.pytorch.log_model(model, artifact_path=f"models/checkpoint_epoch_{epoch}")

            # if test_stats['acc1'] > best_acc:
            #     best_acc = test_stats['acc1']
            #     record_acc = best_acc
            #     record_precision = precision
            #     record_recall = recall
            #     record_f1 = f1
            #     record_epoch = epoch
            #     print(f"New best accuracy: {best_acc:.2f}%. Saving the model...")
            #     model_best_weight = copy.deepcopy(model.state_dict())
                

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                record_acc = test_stats['acc1']
                record_precision = precision
                record_recall = recall
                record_f1 = f1
                record_epoch = epoch
                print(f"New best F1 score: {best_f1:.2f}%. Saving the model...")
                model_best_weight = copy.deepcopy(model.state_dict())
        total_time = time.time() - start_time
        
        mlflow.log_metric('total_time', total_time)
        print(f"Traning best accuracy: {record_acc:.2f}%",f"precision: {record_precision:.2f}%",f"recall: {record_recall:.2f}%",f"f1: {record_f1:.2f}%","epoch:",record_epoch)

        model.load_state_dict(model_best_weight)
        mlflow.pytorch.log_model(model, artifact_path=f"models/epoch_{record_epoch}")
        val_folder = os.path.join(args.data_path, "val")
        output_folder = args.zip_file_name + "gradcam_results"  # 결과 폴더 (없으면 내부에서 생성됨)

        # 대상 layer 선택 (EfficientNet의 경우 model.blocks[-1] 내 마지막 Conv2d)
        conv_layers = []
        if "efficientnet" in args.model_name.lower():
            for m in model.blocks[-1].modules():
                if isinstance(m, nn.Conv2d):
                    conv_layers.append(m)
            if len(conv_layers) > 0:
                target_layer = conv_layers[-1]
            else:
                target_layer = model.features[-1]
        elif "mobilenet" in args.model_name.lower():
            # MobileNet v2의 경우 features의 마지막 부분에서 Conv2d 찾기
            for m in model.blocks[-1].modules():
                if isinstance(m, nn.Conv2d):
                    conv_layers.append(m)
            if len(conv_layers) > 0:
                target_layer = conv_layers[-1]
            else:
                # 대안으로 conv_head 사용
                target_layer = model.conv_head

        elif "resnet" in args.model_name.lower():
            for m in model.layer4.modules():
                if isinstance(m, nn.Conv2d):
                    conv_layers.append(m)
            if len(conv_layers) > 0:
                target_layer = conv_layers[-1]
            else:
                target_layer = model.features[-1]

        elif "deit" in args.model_name.lower(): 
            target_layer = None
        print("Selected target layer for Grad-CAM:", target_layer)

        # gradcam.py에 정의된 process_val_images() 호출
        # ※ process_val_images()는 각 이미지 처리 시 CSV에
        #    [파일명, 원본 레이블 (클래스 폴더 이름), 예측 클래스, 신뢰도] 정보를 기록하도록 되어 있음.
        process_val_images(model, target_layer, device,args,
                           val_folder=val_folder,
                           output_folder=output_folder)
        print("Grad-CAM results have been saved in:", output_folder)
        zip_filename = args.zip_file_name
        shutil.make_archive(zip_filename, 'zip', output_folder)
        mlflow.log_artifact(zip_filename+ ".zip", artifact_path="gradcam_results")
        print("Grad-CAM results ZIP file has been uploaded as an MLflow artifact.")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
