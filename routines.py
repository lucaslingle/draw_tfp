import tensorflow_datasets as tfds
import numpy as np


def train(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):
        elbo, step, summary = model.train(sess, batch)
        print('Global Step {}, ELBO {}'.format(step, elbo))

        _ = callbacks['tensorboard'](step, summary)
        _ = callbacks['checkpointing'](step)


def evaluate(dataset, sess, model, callbacks):
    elbo_sum = 0.0
    batch_count = 0

    for batch in tfds.as_numpy(dataset):

        elbo = model.evaluate(sess, batch)
        elbo_sum += elbo
        batch_count += 1

    elbo_avg = elbo_sum / float(batch_count)
    print('Test ELBO: {}'.format(elbo_avg))


def generate(dataset, sess, model, callbacks):
    gen_xs = model.generate(sess, num_samples=64)
    gen_xs = gen_xs[:,-1,:,:,:] # final frame only

    print('saving images...')
    rows = []
    for i in range(0, 8):
        row = []
        for j in range(0, 8):
            row.append(gen_xs[8 * i + j])
        row = np.concatenate(row, axis=1)
        rows.append(row)

    img = np.concatenate(rows, axis=0)

    fp = callbacks['save_png'](img)
    print(fp)


def reconstruct(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):
        batch_size = batch.shape[0]
        xs = batch
        recon_xs = model.reconstruct(sess, batch)
        recon_xs = recon_xs[:,-1,:,:,:] # final frame only

        print('saving images...')
        columns = []
        for i in range(0, batch_size):
            col = []
            col.append(xs[i])
            col.append(recon_xs[i])
            col = np.concatenate(col, axis=0)
            columns.append(col)

        img = np.concatenate(columns, axis=1)

        fp = callbacks['save_png'](img)
        print(fp)

        break


def generate_gif(dataset, sess, model, callbacks):
    # creates a gif showing the generative process 'drawing' new data.
    # generated samples are arranged in a grid, so that many examples can be visualized simultaneously.

    def make_image_grid(imgs, grid_height, grid_width):
        assert imgs.shape[0] == grid_height * grid_width
        rows = []
        for i in range(0, grid_height):
            row = []
            for j in range(0, grid_width):
                img = imgs[grid_height * i + j]
                row.append(img)
            row = np.concatenate(row, axis=1)
            rows.append(row)
        rows = np.concatenate(rows, axis=0)
        return rows

    S = 8    
    X = model.generate(sess, num_samples=(S * S))

    print('saving images...')
    # make list of frames for the gif
    img_frames = []
    # include the starting images
    xs0 = np.zeros(dtype=np.float32, shape=((S*S), model.img_height, model.img_width, model.img_channels))
    img = make_image_grid(xs0, S, S)
    img_frames.append(img)

    # loop over timesteps and make grid for each
    for t in range(0, model.num_timesteps):
        xs = X[:,t,:,:,:]
        img = make_image_grid(xs, S, S)
        img_frames.append(img)

    fp = callbacks['save_gif'](img_frames)
    print(fp)


def reconstruct_gif(dataset, sess, model, callbacks):

    # creates a gif showing the inference model and generative model reconstucting real examples.
    # generated samples are arranged in a grid, so that many examples can be visualized simultaneously.

    def make_image_grid(imgs, grid_height, grid_width):
        assert imgs.shape[0] == grid_height * grid_width
        rows = []
        for i in range(0, grid_height):
            row = []
            for j in range(0, grid_width):
                img = imgs[grid_height * i + j]
                row.append(img)
            row = np.concatenate(row, axis=1)
            rows.append(row)
        rows = np.concatenate(rows, axis=0)
        return rows

    S = 8

    for batch in tfds.as_numpy(dataset):
        batch_size = batch.shape[0]
        if S * S < batch_size:
            S = int(np.floor(np.sqrt(batch_size)))

        X_orig = batch
        X_recon = model.reconstruct(sess, batch)
        X_recon = X_recon[:, 0:(S*S), :, :, :]  # final frame only

        X_orig_grid = make_image_grid(X_orig, S, S)

        print('saving images...')
        # make list of frames for the gif
        img_frames = []
        # include the starting images
        xs0 = np.zeros(dtype=np.float32, shape=((S * S), model.img_height, model.img_width, model.img_channels))
        img_l = X_orig_grid
        img_c = np.zeros(dtype=np.float32, shape=(S * model.img_height, model.img_width, model.img_channels))
        img_r = make_image_grid(xs0, S, S)
        img = np.concatenate([img_l, img_c, img_r], axis=1)
        img_frames.append(img)

        # loop over timesteps and make grid for each
        for t in range(0, model.num_timesteps):
            xs = X_recon[:, t, :, :, :]
            img_l = X_orig_grid
            img_c =np.zeros(dtype=np.float32, shape=(S * model.img_height, model.img_width, model.img_channels))
            img_r = make_image_grid(xs, S, S)
            img = np.concatenate([img_l, img_c, img_r], axis=1)
            img_frames.append(img)

        fp = callbacks['save_gif'](img_frames)
        print(fp)

        break
