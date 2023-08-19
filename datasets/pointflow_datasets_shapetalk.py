import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import tqdm


class Uniform15KPC(Dataset):
    def __init__(self, pc_files, tr_sample_size=10000,
                 te_sample_size=10000, scale=1.,
                 normalize_per_shape=False, random_subsample=False,
                 normalize_std_per_axis=False, recenter_per_shape=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3):

        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []


        # We changed the code to work with a list of pointclouds obeying the format used in ShapeTalk
        for file_name in tqdm.tqdm(pc_files):
            tokens = file_name.split('/')
            cate_idx = tokens[-3]
            dataset_id = tokens[-2]
            mid = tokens[-2] + '/' + tokens[-1][:-len('.npz')]
            try:
                data = np.load(file_name, allow_pickle=True)['output'].item()['pntcloud']
            except:
                data = np.load(file_name)['pointcloud']

            point_cloud = data[:, 0:3].astype(np.float32)
            point_cloud = point_cloud[:15000]  # consider changing it to variable size indicated by the actual pre-sampled data
            assert point_cloud.shape[0] == 15000

            self.all_points.append(point_cloud[np.newaxis, ...])
            self.cate_idx_lst.append(cate_idx)
            self.all_cate_mids.append((dataset_id, mid))
        self.pc_files = pc_files


        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # edit from ShapeTalk
        self.pc_files = [self.pc_files[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.recenter_per_shape = recenter_per_shape
        if all_points_mean is not None and all_points_std is not None and not self.recenter_per_shape:
            # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.recenter_per_shape:  # per shape center
            # TODO: bounding box scale at the large dim and center
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)
            ) / 2
            self.all_points_std = np.amax((
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)
            ), axis=-1).reshape(B, 1, 1) / 2
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(
                axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(
                -1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    -1).std(axis=0).reshape(1, 1, 1)
        if True: # NOTE: make sure to change!!!!!!!!!!!!!!!!!!
            self.all_points = (self.all_points - self.all_points_mean)
        else:
            self.all_points = (self.all_points - self.all_points_mean) / \
                self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def get_pc_stats(self, idx):
        if self.recenter_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), \
            self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + \
                          self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'tr_points': tr_out,
            'te_points': te_out,
            'mean': m, 'std': s, 'cate_idx': cate_idx,
            'sid': sid, 'mid': mid,
            'display_axis_order': self.display_axis_order
        }


class ShapeModelNet15kPointClouds(Uniform15KPC):
    def __init__(self, pc_files, tr_sample_size=10000,
                 te_sample_size=2048,
                 scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False, recenter_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):

        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size


        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeModelNet15kPointClouds, self).__init__(
            pc_files,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            recenter_per_shape=recenter_per_shape,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(cfg, args):

    import pandas as pd
    import os.path as osp

    def model_to_file_name(df, model_name_col, object_class_col, dataset_col, top_dir, ending):
        def join_parts(row):
            return osp.join(top_dir, row[object_class_col], row[dataset_col], row[model_name_col] + ending)
        return df.apply(join_parts, axis=1)

    split_df = pd.read_csv(args.split_file)
    datasets = dict()

    # FOR DEBUGGING!
    # if args.chairs_only:
    #     print("DEBUGGING! KEEPING ONLY CHAIRS")
    #     split_df = split_df[split_df['shape_class'] == 'chair']

    for split in split_df.split.unique():
        print(split)
        ndf = split_df[split_df.split == split].copy()
        pc_file_names = model_to_file_name(ndf, 'model_name', 'shape_class', 'dataset', cfg.data_dir, '.npz').values

        dataset = ShapeModelNet15kPointClouds(
            pc_file_names,
            tr_sample_size=cfg.tr_max_sample_points,
            te_sample_size=cfg.te_max_sample_points,
            scale=cfg.dataset_scale,
            normalize_per_shape=cfg.normalize_per_shape,
            normalize_std_per_axis=cfg.normalize_std_per_axis,
            recenter_per_shape=cfg.recenter_per_shape,
            random_subsample=True)
        datasets[split] = dataset
    return datasets


def get_data_loaders(cfg, args):
    datasets = get_datasets(cfg, args)
    test_only = True

    loaders = dict()
    if test_only:
        for split in datasets:
            loader = data.DataLoader(
                dataset=datasets[split],
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
                worker_init_fn=init_np_seed,
                pin_memory=False
            )
            loaders[split + '_loader'] = loader

    if not test_only:
        raise NotImplementedError

    return loaders
