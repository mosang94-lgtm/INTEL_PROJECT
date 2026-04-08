# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import random
import argparse
from functools import partial

from src.Models import Unet
from src.Train import Trainer
from src.Evaluation import Evaluation
from src.Utils import FocalLoss
from torch.optim.lr_scheduler import StepLR
import albumentations as A



if __name__ == "__main__":
    ## weighted loss
    weights = torch.tensor([100,1], dtype=torch.float32)
    # weights = 1.0 / weights weight=weights.cuda()
    weights = weights / weights.sum()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train',  help='train')
    parser.add_argument('--eval', help='evaluation')
    parser.add_argument('--task', help='damage vs part')
    parser.add_argument('--method', help='multi : multi models, single : single model')
    parser.add_argument('--label', help="all, 2,3,4,5")
    parser.add_argument('--cls', type = int, help = "n_class")
    parser.add_argument('--dataset', help = "val vs test")
    parser.add_argument('--weight_file', help = "weight file name")

    arg = parser.parse_args()


    ## device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    if device == 'cuda':
        print(f'gpu device num: {torch.cuda.current_device()}')

    ## model

    if arg.task == 'damage':
        n_cls = 2
    elif arg.eval:
        n_cls = 16
    elif arg.cls is not None:
        n_cls = arg.cls
    else:
        print("[오류] --task 또는 --cls 인자가 필요합니다.")
        print("  damage 학습 예시 : python main.py --train train --task damage --label all")
        print("  part   학습 예시 : python main.py --train train --task part --cls 16")
        print("  평가   예시      : python main.py --eval y --task damage --dataset val")
        exit(1)

    model = Unet(encoder="resnet34",pre_weight='imagenet',num_classes=n_cls)

    ## set seed
    def set_seed(seed:int):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    set_seed(1230)


    ## model load
    def load_model(model, weight_path, strict):
        model = model.to(device)
        try:
            model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device), weights_only=False),strict=strict)
            return model
        except:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device(device), weights_only=False),strict=strict)
            return model


    ## train
    if arg.train:
        # train damage
        if (arg.label == "all") & (arg.task == "damage"):
            label_schme = ["Scratched","Separated","Crushed","Breakage"]
            for i in range(4):
                trainer = Trainer(
                            ails = f"{arg.task}",
                            train_dir = f"../data/datainfo/{arg.task}_{label_schme[i]}_train.json",
                            val_dir = f"../data/datainfo/{arg.task}_val.json",
                            img_base_path = '../data/Dataset/1.원천데이터/damage',
                            size = 256,
                            model = model,
                            label = i,
                            n_class = n_cls,
                            optimizer = torch.optim.Adam,
                            criterion = torch.nn.CrossEntropyLoss(),
                            epochs = 100,
                            batch_size = 64,
                            encoder_lr = 1e-06,
                            decoder_lr = 3e-04,
                            weight_decay = 0,
                            device = device)
                trainer.train()
        # train damage one label
        elif (arg.task == 'demage') & (arg.method == 'multi'):
            trainer = Trainer(
                        ails = f"{arg.task}_label{arg.label}",
                        train_dir = f"../data/datainfo/{arg.task}_trainsample.json",
                        val_dir = f"../data/datainfo/{arg.task}_valsample.json",
                        img_base_path = '../data/Dataset/1.원천데이터/damage_part',
                        size = 256,
                        model = model,
                        label = arg.label,
                        n_class = n_cls,
                        optimizer = torch.optim.Adam,
                        criterion = torch.nn.CrossEntropyLoss(),
                        epochs = 70,
                        batch_size = 32,
                        encoder_lr = 1e-07,
                        decoder_lr = 1e-06,
                        weight_decay = 0,
                        device = device)
            trainer.train()

        # train part_ver2
        else:

            transform = A.Compose([
                            A.RandomRotate90(p=0.3),
                            A.CoarseDropout(p=0.3, num_holes_range=(1, 8), hole_height_range=(32, 32), hole_width_range=(32, 32)),
                            A.Resize(height=256, width=256)])

            scheduler = partial(StepLR, step_size=10, gamma=0.9)
            trainer = Trainer(
                ails = f"{arg.task}",
                train_dir = f"../data/datainfo/{arg.task}_train.json",
                val_dir = f"../data/datainfo/{arg.task}_val.json",
                img_base_path = '../data/Dataset/1.원천데이터/damage_part',
                size = 256,
                model = model,
                label = None,
                n_class = arg.cls,
                optimizer = torch.optim.Adam,
                criterion = torch.nn.CrossEntropyLoss(),
                epochs = 57,
                batch_size = 32,
                encoder_lr = 1e-06,
                decoder_lr = 3e-04,
                weight_decay = 1e-02,
                device = device,
                transform = transform,
                lr_scheduler = scheduler,
                start_epoch = None)
            trainer.train()


    if arg.eval:
        set_seed(12)
        # evaluation
        if arg.task == 'damage':
            evaluation = Evaluation(
                        eval_dir = f"../data/datainfo/damage_{arg.dataset}.json",
                        size = 256,
                        model = model,
                        weight_paths = ["../data/weight/"+n for n in ["Unet_damage_label0_start:2026-04-07 15:03:09 KST+0900_2_epoch_IoU_2e+01.pt","Unet_damage_label1_start:2026-04-07 15:01:00 KST+0900_2_epoch_IoU_0.9.pt","Unet_damage_label2_start:2026-04-07 15:02:24 KST+0900_2_epoch_IoU_0e+00.pt","Unet_damage_label3_start:2026-04-07 15:03:41 KST+0900_2_epoch_IoU_0.7.pt"]],
                        device = device,
                        batch_size = 64,
                        ails = f"../data/result_log/[{arg.task}]_{arg.dataset}_evaluation_log.json",
                        criterion = torch.nn.CrossEntropyLoss(),
                        img_base_path = "../data/Dataset/1.원천데이터/damage"
            )
            evaluation.evaluation()
        else:
            model = Unet(encoder="resnet34",pre_weight='imagenet',num_classes=n_cls)
            if arg.weight_file:
                weight_path = f"../data/weight/{arg.weight_file}"
            else:
                weight_path = "../data/weight/[PART]Unet.pt"
            evaluation = Evaluation(
                        eval_dir = f"../data/datainfo/part_{arg.dataset}.json",
                        size = 256,
                        model = model,
                        weight_paths = [weight_path],
                        device = device,
                        batch_size = 64,
                        ails = f"../data/result_log/[{arg.task}]_{arg.dataset}_evaluation_log.json",
                        criterion = torch.nn.CrossEntropyLoss(),
                        img_base_path = "../data/Dataset/1.원천데이터/damage_part"
            )
            evaluation.evaluation()
