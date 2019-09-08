import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

from draw import DRAW
from data import get_dataset
import callbacks as calls
import routines

flags = tf.app.flags


flags.DEFINE_enum("mode", 'train', ['train', 'eval', 'generate', 'reconstruct', 'generate_gif', 'reconstruct_gif'], 
    "mode: one of train, eval, generate, reconstruct, generate_gif, reconstruct_gif")

flags.DEFINE_enum("dataset", 'mnist', ['mnist', 'celeb_a', 'cifar10', 'omniglot', 'svhn_cropped'], "dataset: which dataset to use")
flags.DEFINE_integer("img_height", 32, "img_height: height to scale images to, in pixels")
flags.DEFINE_integer("img_width", 32, "img_width: width to scale images to, in pixels")
flags.DEFINE_integer("img_channels", 1, "img_channels: number of image channels")

flags.DEFINE_integer("batch_size", 100, "batch_size: number of examples per minibatch")
flags.DEFINE_integer("read_dim", 2, "read_dim: size of gaussian kernel grid for reading")
flags.DEFINE_integer("write_dim", 5, "write_dim: size of gaussian kernel grid for writing")

flags.DEFINE_integer("num_timesteps", 64, "num_timesteps: number of steps for drawing image")
flags.DEFINE_integer("z_dim", 100, "z_dim: dimension of latent variable z")
flags.DEFINE_integer("encoder_hidden_dim", 256, "encoder_hidden_dim: number of hidden units in encoder LSTM")
flags.DEFINE_integer("decoder_hidden_dim", 256, "decoder_hidden_dim: number of hidden units in decoder LSTM")
flags.DEFINE_float("init_scale", 0.10, "init_scale: scale for weight init")
flags.DEFINE_float("lr", 0.001, "lr: learning rate for Adam optimizer")

flags.DEFINE_string("summaries_dir", 'tensorboard_logs/', "summaries_dir: directory for tensorboard logs")
flags.DEFINE_string("output_dir", 'output/', "output_dir: directory for visualizations")

flags.DEFINE_string("checkpoint_dir", 'checkpoints/', "checkpoint_dir: directory for saving model checkpoints")
flags.DEFINE_string("load_checkpoint", '', "load_checkpoint: checkpoint directory or checkpoint to load")
flags.DEFINE_integer("checkpoint_frequency", 500, "checkpoint_frequency: frequency to save checkpoints, measured in global steps")

flags.DEFINE_integer("epochs", 10, "epochs: number of epochs to train for. ignored if mode is not 'train'")

FLAGS = flags.FLAGS


def main(_):

    ## hyperparams
    hps = tf.contrib.training.HParams(
        batch_size = FLAGS.batch_size,
        img_height = FLAGS.img_height,
        img_width = FLAGS.img_width,
        img_channels = FLAGS.img_channels,
        num_timesteps = FLAGS.num_timesteps,
        z_dim = FLAGS.z_dim,
        encoder_hidden_dim = FLAGS.encoder_hidden_dim,
        decoder_hidden_dim = FLAGS.decoder_hidden_dim,
        read_dim = FLAGS.read_dim,
        write_dim = FLAGS.write_dim,
        init_scale = FLAGS.init_scale,
        lr = FLAGS.lr,
        epochs = FLAGS.epochs)

    ## dataset
    ds_train, ds_test = get_dataset(name=FLAGS.dataset, hps=hps)

    ## model and session
    model = DRAW(hps)
    sess = tf.Session()

    ## tensorboard
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    ## checkpointing
    saver = tf.train.Saver()

    ## init op
    init_op = tf.global_variables_initializer()
    _ = sess.run(init_op)

    ## restoring
    if FLAGS.load_checkpoint != '':
        saver.restore(sess, FLAGS.load_checkpoint)

    ## helper functions for the various modes supported by this application
    mode_to_routine = {
        'train': routines.train,
        'eval': routines.evaluate,
        'generate': routines.generate,
        'reconstruct': routines.reconstruct,
        'generate_gif': routines.generate_gif
    }
    routine = mode_to_routine[FLAGS.mode]

    ## rather than pass around tons of arguments,
    #  just use callbacks to perform the required functionality
    if FLAGS.mode == 'train':
        checkpoint_dir = FLAGS.checkpoint_dir
        checkpoint_frequency = FLAGS.checkpoint_frequency
        callbacks = {
            'tensorboard': calls.tensorboard(train_writer), 
            'checkpointing': calls.checkpointing(sess, saver, checkpoint_dir, checkpoint_frequency)
        }
        routine(ds_train, sess, model, callbacks)

    elif FLAGS.mode == 'eval':
        callbacks = {}
        routine(ds_test, sess, model, callbacks)

    else:
        output_dir = FLAGS.output_dir
        callbacks = {
            'save_png': calls.save_png(output_dir),
            'save_gif': calls.save_gif(output_dir)
        }
        routine(ds_train, sess, model, callbacks)


if __name__ == '__main__':
    tf.app.run()
