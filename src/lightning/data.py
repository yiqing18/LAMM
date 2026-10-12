import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed
import torch

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
)
import pdb

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm
from src.datasets.vissar import VISSARDataset
from src.datasets.sampler import RandomConcatSampler



class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE 
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT  
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH  
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT    
        self.pair_list_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT   #
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }
        
        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT
        
        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        # except AssertionError as ae:
        except (AssertionError, RuntimeError) as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")
        self.world_size = 1
        self.rank = 0

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                self.pair_list_path,
                mode='train',
                min_overlap_score=self.min_overlap_score_train)
            # setup multiple (optional) validation subsets
            self.val_dataset = self._setup_dataset(
                self.val_data_root,
                self.val_npz_root,
                self.val_list_path,
                self.val_intrinsic_path,
                mode='val',
                min_overlap_score=self.min_overlap_score_test)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                self.val_intrinsic_path,
                mode='test',
                min_overlap_score=self.min_overlap_score_test)
            # logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       pair_list_path,
                       mode='train',
                       min_overlap_score=0.):
        """ Setup train / val / test set"""
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        # pdb.set_trace()
        if mode == 'train':
            local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
        else:
            local_npz_names = npz_names
        # logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')
        
        dataset_builder = self._build_concat_dataset_parallel \
                            if self.parallel_load_data \
                            else self._build_concat_dataset
        return dataset_builder(data_root, local_npz_names, split_npz_root,pair_list_path,
                                mode=mode, min_overlap_score=min_overlap_score)

    def _build_concat_dataset(
        self,
        data_root,
        npz_names,
        npz_dir,
        pair_dir,
        mode,
        min_overlap_score=0.,
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        for npz_name in tqdm(npz_names,
                             desc=f'[rank:{self.rank}] loading {mode} datasets',
                             disable=int(self.rank) != 0):
            if data_source == 'VISSAR':
                datasets.append(
                    VISSARDataset(data_root,
                                   npz_dir,
                                   npz_name,
                                   pair_dir,
                                   mode=mode,
                                   min_overlap_score=min_overlap_score,
                                   augment_fn=augment_fn))
            else:
                raise NotImplementedError()
        return ConcatDataset(datasets)
    
    def _build_concat_dataset_parallel(
        self,
        data_root,
        npz_names,
        npz_dir,
        pair_dir,
        mode,
        min_overlap_score=0.,
    ):
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        with tqdm_joblib(tqdm(desc=f'[rank:{self.rank}] loading {mode} datasets',
                              total=len(npz_names), disable=int(self.rank) != 0)):
            if data_source == 'VISSAR':
                datasets = Parallel(n_jobs=math.floor(len(os.sched_getaffinity(0)) * 0.9 / comm.get_local_size()))(
                    delayed(lambda x: _build_dataset(
                        VISSARDataset,
                        data_root,
                        npz_dir,
                        name,
                        pair_dir,
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        augment_fn=augment_fn))(name)
                    for name in npz_names)
            else:
                raise ValueError(f'Unknown dataset: {data_source}')
        return ConcatDataset(datasets)

    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        assert self.data_sampler in ['scene_balance']
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                          self.n_samples_per_subset,
                                          self.subset_replacement,
                                          self.shuffle, self.repeat, self.seed)
        else:
            sampler = None
        # pdb.set_trace()
        dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        return dataloader

    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if not isinstance(self.val_dataset, abc.Sequence):
            # sampler = DistributedSampler(self.val_dataset, shuffle=False)
            return DataLoader(self.val_dataset,  **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                # sampler = DistributedSampler(dataset, shuffle=False)
                dataloaders.append(DataLoader(dataset, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        # sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
