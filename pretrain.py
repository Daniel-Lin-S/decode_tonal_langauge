import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch
import os
import yaml
import numpy as np

from models.foundation import CBraMod, PretrainTrainer
from utils.utils import set_seeds


models_choices = ['CBraMod']

def main():
    parser = argparse.ArgumentParser(
        description='pre-train foundation model on ECoG data'
    )
    # -------- General Parameters --------
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--cuda', type=int, default=0,
        help='Cuda device id (default: 0)'
        'Automatically set device to cpu if no GPU is available.'
    )
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Use DataParallel to compute on multiple GPUs'
    )
    parser.add_argument(
        '--loader_num_workers', type=int, default=8,
        help='Number of workers for the DataLoader (default: 8)'
    )
    # -------- Training Parameters --------
    parser.add_argument(
        '--model', type=str, default='CBraMod',
        choices=models_choices,
        help='The model to use for pretraining (default: CBraMod)'
    )
    parser.add_argument(
        '--epochs', type=int, default=40,
        help='number of epochs for training (default: 5)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size for training (default: 128)'
    )
    parser.add_argument(
        '--need_mask', type=bool, default=True,
        help='Whether to use masked input for training (default: True)'
        'If False, the model simply reconstructs the input.'
    )
    parser.add_argument(
        '--mask_ratio', type=float, default=0.5,
        help='Mask ratio for the masked input (default: 0.5)'
    )
    # -------- Optimizer Parameters --------
    parser.add_argument(
        '--lr', type=float, default=5e-4,
        help='learning rate of the optimiser (default: 5e-4)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=5e-2,
        help='weight decay of the optimiser (default: 5e-2)'
    )
    parser.add_argument(
        '--clip_value', type=float, default=1.,
        help='gradient clipping value for the optimiser (default: 1.0)'
    )
    parser.add_argument(
        '--lr_scheduler', type=str, default='CosineAnnealingLR',
        choices=['CosineAnnealingLR', 'ExponentialLR', 'StepLR', 'MultiStepLR', 'CyclicLR'],
        help='The learning rate scheduler to use (default: CosineAnnealingLR)'
    )
    # -------- I/O settings --------
    parser.add_argument(
        '--sample_path', type=str, required=True,
        help='Path to the npy file containing the samples for pre-training.'
    )
    parser.add_argument(
        '--model_ckpt_path', type=str, required=True,
        help='Path to the pre-trained weights of the model.'
    )
    parser.add_argument(
        '--model_configs', type=str, default='configs/model_configs.yaml',
        help='Path to the configuration file for the models.'
        'Must be a YAML file, and have a key of the model name '
        'with the model parameters as a dictionary.'
    )
    parser.add_argument(
        '--model_dir', type=str, default='checkpoints',
        help='Directory to save the model checkpoints '
        '(default: checkpoints)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='tensorboards/pretrain',
        help='Directory to save TensorBoard logs '
    )
    parser.add_argument(
        '--log_interval', type=int, default=10,
        help='Interval (in batches) to log training progress '
        'if set to 0, will only log every epoch.'
    )

    params = parser.parse_args()

    # ------ value checks ------
    if not os.path.exists(params.model_configs):
        raise FileNotFoundError(
            f"Model configuration file '{params.model_configs}' does not exist."
        )

    # load configuration
    with open(params.model_configs, 'r') as f:
        model_configs = yaml.safe_load(f)

    if params.model not in model_configs:
        raise ValueError(
            f"Model configuration for '{params.model}' not found in {params.model_configs}."
        )

    print("Parameters: ", params)

    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    set_seeds(params.seed)

    samples = np.load(params.sample_path)

    samples_tensor = torch.from_numpy(samples).float()
    dataset = TensorDataset(samples_tensor)

    print('Number of samples in the pretraining dataset: ',
          len(dataset))

    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=8,
        shuffle=True,
    )

    device = torch.device(
        f"cuda:{params.cuda}"
        if torch.cuda.is_available() else "cpu"
    )

    if params.model == 'CBraMod':
        model = CBraMod(**model_configs['CBraMod'])

    else:
        raise ValueError(
            f"Model '{params.model}' is not supported."
            f" Supported models: {models_choices}"
        )
    
    model.load_state_dict(
        torch.load(params.model_ckpt_path, map_location=device)
    )

    trainer = PretrainTrainer(
        params, data_loader, model,
        log_interval=params.log_interval
    )
    trainer.train()

    print('Training completed.')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
