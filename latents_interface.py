import multiprocessing
import os
import yaml
import time
import torch
import torch.nn as nn
import argparse
import importlib
from torch.backends import cudnn
from shutil import copy2
from pprint import pprint

from ShapeGF.trainers.utils.vis_utils import visualize_point_clouds_3d 

from six.moves import cPickle
from tqdm import tqdm
import numpy as np
from PIL import Image
import dill as pickle
# from multiprocessing import Pool, Process
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


def safe_makedirs(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        print('Folder exists already.')        

def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()

def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')

    # Resume:
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    # Evaluation split
    parser.add_argument('--eval_split', default='val', type=str,
                        help="The split to be evaluated.")
    parser.add_argument('--chairs_only', default=False, action='store_true')    
    parser.add_argument('--split_file', default='', type=str)
    parser.add_argument('--num_chunks', default='4', type=int)
    parser.add_argument('--chunk_index', type=int)
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # Currently save dir and log_dir are the same
    config.log_name = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.save_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.log_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    safe_makedirs(config.log_dir+'/config')
    copy2(args.config, config.log_dir+'/config')
    return args, config



class SGFWrapper(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.trainer_lib = importlib.import_module(cfg.trainer.type)
        self.trainer = self.trainer_lib.Trainer(cfg, args)
    
        if args.distributed:  # Multiple processes, single GPU per process
            def wrapper(m):
                return nn.DataParallel(m)
            self.trainer.multi_gpu_wrapper(wrapper)
        self.trainer.resume(args.pretrained)

        # below adapted from mesh_extraction.py 
        points_batch_size=100000 
        t=0.005
        self.sigma = self.trainer.sigmas[-1] 
        self.sigma = torch.tensor(self.sigma)
        # model = ModelWrapper(self.trainer, t, points_batch_size)
        # self.model_mesh = Generator3D(model, points_batch_size=points_batch_size, threshold=0., resolution0=32)

        pass
    
    def write_ply_triangle(self, name, vertices, triangles):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face "+str(len(triangles))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
        for ii in range(len(triangles)):
            fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
        fout.close()

    def get_z(self):
        # get the dataset
        data_lib = importlib.import_module(self.cfg.data.type)
        dataloaders = data_lib.get_data_loaders(self.cfg.data, args) 
        zs, _, _ = self.trainer.extract_latents_from_loaders(dataloaders)
        print('done')
        return zs


    def decode_visualize(self, id_, z, save_output=False, output_dir=None, verbose=False):
        latent = torch.tensor(z[id_]).unsqueeze(0).to('cuda:0')
        final_pc, _ = self.trainer.langevin_dynamics(latent)
        gradient_on_query = self.trainer.extract_gradient_on_query_from_latent(latent, n_cubes=64)
        gradient_on_query = gradient_on_query[-1]
        mesh_vertices, mesh_faces = self.trainer_lib.Trainer.extract_mesh_from_gradient(gradient_on_query, level_set=60)
        
        if save_output :
            assert output_dir is not None

            pc_path = output_dir+"/"+id_+".npz"
            if not os.path.exists(os.path.dirname(pc_path)):
                safe_makedirs(os.path.dirname(pc_path))
            np.savez(pc_path, pc=final_pc.cpu().numpy().squeeze())

            img_path = output_dir+"/"+id_+".png"
            if not os.path.exists(os.path.dirname(img_path)):
                safe_makedirs(os.path.dirname(img_path))
            img = visualize_point_clouds_3d([final_pc.cpu().squeeze()], ['final_pc']) 
            img = Image.fromarray(img[:3].transpose(1,2,0))
            img.save(img_path)

            mesh_path = output_dir+"/"+id_+'.ply'
            if not os.path.exists(os.path.dirname(img_path)):
                safe_makedirs(os.path.dirname(img_path))
            self.write_ply_triangle(mesh_path, mesh_vertices, mesh_faces)

            if verbose:
                print(f"Saved mesh at {mesh_path}")
                print(f"Saved pointcloud at {pc_path}")
                print(f"Saved pointcloud visualization at {img_path}") 

        return (mesh_vertices, mesh_faces) , final_pc.squeeze().cpu().detach().numpy()  # img_path, pc_path


    def eval_z_ian(self, z, save_output=False, output_dir=None, verbose=False):
                        
        if output_dir is not None and not os.path.exists(output_dir): 
            os.makedirs(dir)            
        
        meshes = dict()
        pcs = dict()
                
        for id_ in tqdm(z):
            mesh, pc = self.decode_visualize(id_, z, save_output=save_output, output_dir=output_dir, verbose=verbose) 
            meshes[id_] = mesh
            pcs[id_] = pc
        
        return meshes, pcs

    @torch.no_grad()
    def eval_z(self, z, compute_mesh=False, npc_points=4096, save_output=False, output_dir=None, verbose=False, 
               skip_existing=False, return_results=True):
        ## todo fix device to be arbitrary

        original_pts = self.trainer.cfg.inference.num_points
        self.trainer.cfg.inference.num_points = npc_points
        
        ids = []
        zs = [] 
        for id_ in sorted(list(z.keys()), reverse=True):
            ids.append(id_)
            zs.append(z[id_])
        
        meshes = dict()
        pcs = dict()
        
        for i in tqdm(range(len(zs))):
            id_ = ids[i]
            z = zs[i]         
                                            
            if skip_existing and output_dir is not None:
                pc_path = output_dir + "/" + id_ + ".npz"                
                if os.path.exists(pc_path):
                    if verbose:
                        print(f'skipping {id_}')
                    continue
                        
            latent = torch.tensor(z).unsqueeze(0).to('cuda:0')
            final_pc, _ = self.trainer.langevin_dynamics(latent)
            final_pc = final_pc.cpu().numpy().squeeze()
            
            if return_results:
                pcs[id_] = final_pc
                            
            if save_output:
                pc_path = output_dir + "/" + id_ + ".npz"
                if not os.path.exists(os.path.dirname(pc_path)):
                    os.makedirs(os.path.dirname(pc_path))
                np.savez(pc_path, pc=final_pc)
            
            if compute_mesh:
                gradient_on_query = self.trainer.extract_gradient_on_query_from_latent(latent, n_cubes=64)
                gradient_on_query = gradient_on_query[-1]
                mesh_vertices, mesh_faces = self.trainer_lib.Trainer.extract_mesh_from_gradient(gradient_on_query, level_set=60)
                
                if return_results:
                    meshes[id_] = (mesh_vertices, mesh_faces)
                
                if save_output:
                    mesh_path = output_dir + "/" + id_ +'.ply'                    
                    self.write_ply_triangle(mesh_path, mesh_vertices, mesh_faces)
            
            if verbose:
                print(f'Done with {id_}.')
        
        self.trainer.cfg.inference.num_points = original_pts
        return meshes, pcs

if __name__ == '__main__':
    args, cfg = get_args()
    sgfw = SGFWrapper(cfg, args) 
    with open('SGF-latent-interface-pub.pkl', 'wb') as f:
        pickle.dump(sgfw, f)
    print('dilled interface to SGF-latent-interface-pub.pkl')
