# draw_tfp
Tensorflow Implementation of 'DRAW: A Recurrent Neural Network For Image Generation'

![mnist generation gif](assets/mnist_generation.gif)

## Dependencies
```
tensorflow==1.13.1
tensorflow_probability==0.6.0
tensorflow_datasets==1.0.2
numpy==1.16.4
matplotlib==3.1.0
moviepy==1.0.0
```

## Getting started
#### Step 1.

Install the dependencies. I suggest creating separate python environment for this project.
You can do this easily using virtualenv or Anaconda 3.

After activating the virtual environment, install the dependencies.

A Note for Anaconda 3 users:

<details>
  <summary>click to expand</summary>

```
The tensorflow libraries are not available using 'conda install'. You'll have to get them using pip.

This causes some issues when installing the other dependencies. 

- I observed that installing the other dependencies using conda caused numpy incompatibility with Tensorflow. 
- Tensorflow installs a particular version of numpy (1.16.4) alongside it. 
- Conda does not appear to know about Tensorflow's dependency on this version of numpy.
- Conda overwrites it when you install other dependencies such as matplotlib. 

- As a workaround, you can install all the dependencies via pip, within the conda environment. 
- This solves the problem completely.
```
</details>

#### Step 2. 
Clone this repo, and navigate to it.

#### Step 3. 
Create directories for tensorboard logs, checkpoints, and outputs from the model (such as generated samples).
```
(draw_virtualenv) $ git clone https://github.com/lucaslingle/draw_tfp/
(draw_virtualenv) $ cd draw_tfp
(draw_virtualenv) $ mkdir tensorboard_logs
(draw_virtualenv) $ mkdir checkpoints
(draw_virtualenv) $ mkdir output
```

## Training a model
This implementation supports MNIST, CIFAR-10, and SVHN. 

The default settings are configured with the hyperparameters to train a DRAW model on MNIST.  
You can train a DRAW model on MNIST by running

```
(draw_virtualenv)$ python app.py --dataset=mnist \
--mode=train --epochs=10 \
--checkpoint_dir=checkpoints/mnist_model_1/ --summaries_dir=tensorboard_logs/mnist_model_1/ 
```

It should be done running in less than an hour, which is a 48x speedup over competing implementations.

To train a model on SVHN, here are some hyperparameters that work well:

```
(draw_virtualenv)$ python app.py --dataset=svhn_cropped \
--img_height=32 --img_width=32 --img_channels=3 \
--batch_size=100 --encoder_hidden_dim=512 --decoder_hidden_dim=512 --z_dim=100 --num_timesteps=32 --read_dim=5 --write_dim=5 \
--lr=0.0001 \
--mode=train --epochs=10 \
--checkpoint_dir=checkpoints/svhn_model_1/ --summaries_dir=tensorboard_logs/svhn_model_1/
```

## Generating samples

Once you've trained a model, you can generate samples like so:

```
(draw_virtualenv)$ python app.py --dataset=svhn_cropped	\
--img_height=32 --img_width=32 --img_channels=3 \
--batch_size=100 --encoder_hidden_dim=512 --decoder_hidden_dim=512 --z_dim=100 --num_timesteps=32 --read_dim=5 --write_dim=5 \
--mode=generate_gif \
--checkpoint_dir=checkpoints/svhn_model_1/ --load_checkpoint=checkpoints/svhn_model_1/ \
--output_dir=output/
```

And you'll get a GIF of the model drawing!

![svhn generation gif](assets/svhn_generation.gif)