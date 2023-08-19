# the code is based on occupancy_networks repository
# https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh
import torch.nn as nn
import torch
import trimesh
import argparse
import importlib
import os
import yaml
import numpy as np
from panos_extra_stuff.mesh_extraction.im2mesh.onet.generation import Generator3D

class ModelWrapper(nn.Module):
    def __init__(self, trainer, t, points_batch_size):
        super().__init__()
        self.trainer = trainer
        self.device = None
        self.t = t
        self.points_batch_size = points_batch_size
    def eval_points(self, p, z, sigma=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p = p/0.5
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        z_sigma = torch.cat((z, sigma.expand(z.size(0), 1)), dim=1).to(self.device)
        #print("point range: ", p.max(), p.min())
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # grad = self.trainer.score_net(pi.cuda(), z_sigma.cuda(), opt='field').view(1, -1, 3).norm(keepdim=True, dim=-1).view(-1).cpu()
                grad = self.trainer.score_net(pi.cuda(), z_sigma.cuda()).view(1, -1, 3).norm(keepdim=True, dim=-1).view(-1).cpu()
                #grad = torch.abs(pi.view(1, -1, 3).norm(keepdim=True,dim=-1) - 0.25).float().view(-1)
            occ_hats.append(grad.squeeze(0).detach().cpu())
        occ_hat = torch.cat(occ_hats, dim=0)
        #print("original grad range: ", occ_hat.min(), occ_hat.max())
        occ_hat = - occ_hat + self.t
        #print("update grad range: ", occ_hat.min(), occ_hat.max())
        return occ_hat


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__=='__main__':
    # initialization
    config_dir = "/orion/u/ianhuang/Laser/ShapeGF/configs/recon/shapenet/shapetalk_recon.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.load(f)
    cfg = dict2namespace(config)
    cfg.save_dir= "."

    import ipdb; ipdb.set_trace()
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']

    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, [])
    checkpoint_dir = "path/to/pretrained_model"
    trainer.resume(checkpoint_dir)

    trainer.encoder.eval()
    trainer.score_net.eval()
    sigma = trainer.sigmas[-1] 
    sigma = torch.tensor(sigma)


    # get shape latent code z
    for data in test_loader:
        with torch.no_grad():
            tr_shape = data['tr_points'].cuda().float()
            te_shape = data['te_points'].cuda().float()
            m = data['mean'].float()
            std = data['std'].float()        
            
            sid, mid = data['sid'], data['mid']   
            z_mu, z_sigma = trainer.encoder(tr_shape)
            z = z_mu + 0 * z_sigma
            break


    # extract mesh for latent code z with MISE algorithm 
    # t is the threshold (the threshold might be different for different categories)
    save_path = "."
    t=0.005
    points_batch_size=100000
    model = ModelWrapper(trainer, t, points_batch_size)
    model_mesh = Generator3D(model, points_batch_size=points_batch_size, threshold=0., resolution0=32)
    mesh = model_mesh.generate_from_latent(z.cpu(), sigma.cpu())
    scene = trimesh.scene.Scene(mesh)
    scene.set_camera(angles=(-np.pi/8, -np.pi/4*3, 0))
    png = scene.save_image(resolution=[640, 480])
    file_name = os.path.join(save_path, "surface.png")
    with open(file_name, 'wb') as f:
        f.write(png)
        f.close() 

