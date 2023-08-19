# ShapeGF adapted to work with ShapeTalk & ChangeIt3D

### This codebase is adapted from ShapeGF's original [repository](https://github.com/RuojinCai/ShapeGF)

## Please follow the following setup steps to enable the usage of our pre-trained ShapeGF (SFG) AE with ShapeTalk, within the [ChangeIt3D](https://github.com/optas/changeit3d) codebase.

Below we assume that you have installed the `changeit3d` conda environment as instructed [here](https://github.com/RuojinCai/ShapeGF).

The following dependencies of ShapeGF are **not** included in the changeit3d environment, by default, and you need to install them separately now: 
- pathos
- tensorboardX

To install them in your `changeit3d` conda environment and use a ChangeIt3D network trained with ShapeFG. Do:

```bash
conda activate changeit3d

conda install tensorboardX
conda install -c conda-forge pathos

git clone https://github.com/optas/ShapeGF.git
cd ShapeGF
```

Now, (inside the ShapeGF) repo continue like this:

Download the pretrained checkpoint.
```bash
wget http://download.cs.stanford.edu/orion/changeit3d/shapeGF_ckpt.zip .
unzip shapeGF_ckpt.zip; rm -rf shapeGF_ckpt.zip
```
And run: 
```bash
python latents_interface.py \
    configs/recon/shapenet/shapetalk_public_recon.yaml \
    --pretrained shapeGF_ckpt/epoch_1199_iters_386400.pt 
```

Running the above produces `SGF-latent-interface-pub.pkl` at the top-level directory. Now, given shape latents in an np.array (zs) you can decode them like this:

```python 
import dill as pickle
with open('SGF-latent-interface-pub.pkl', 'wb') as f: 
    sgf = pickle.load(f)

sgf.eval_z(zs, save_output=True, output_dir=OUTPUT_FOLDER)  # OPTION 1: save outputs
outputs = sgf.eval_z(zs) # OPTION 2: returns outputs
```

