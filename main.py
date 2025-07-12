import torch
import argparse
from model import EfficientDet
from config import *
from train import train
from validation import validate
from dataloader import get_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='efficientdet-d0')
    parser.add_argument('-mode', choices=['trainval', 'eval'], default='trainval')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--device', default=0, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if args.cuda else 'cpu')

    model = EfficientDet.from_pretrained(args.model).to(device)

    if args.mode == 'trainval':
        train(model, device, args)
        validate(model, device, args)
    elif args.mode == 'eval':
        validate(model, device, args)

if __name__ == '__main__':
    main()
