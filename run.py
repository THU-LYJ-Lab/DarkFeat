import cv2
import yaml
import argparse
import os
from torch.utils.data import DataLoader

from datasets.gl3d_dataset import GL3DDataset
from trainer import Trainer
from trainer_single_norel import SingleTrainerNoRel
from trainer_single import SingleTrainer


if __name__ == '__main__':
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/nvme2n1/hyz/data/GL3D')
    parser.add_argument('--data_split', type=str, default='comb')
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--job_name', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--start_cnt', type=int, default=0)
    parser.add_argument('--stage', type=int, default=1)
    args = parser.parse_args()

    # load global config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # setup dataloader
    dataset = GL3DDataset(args.dataset_dir, config['network'], args.data_split, is_training=args.is_training)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.stage == 1:
        trainer = SingleTrainerNoRel(config, f'cuda:0', data_loader, args.job_name, args.start_cnt)
    elif args.stage == 2:
        trainer = SingleTrainer(config, f'cuda:0', data_loader, args.job_name, args.start_cnt)
    elif args.stage == 3:
        trainer = Trainer(config, f'cuda:0', data_loader, args.job_name, args.start_cnt)
    else:
        raise NotImplementedError()
        
    trainer.train()

    