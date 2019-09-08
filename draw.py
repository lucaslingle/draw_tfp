import tensorflow as tf
_ = tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_probability as tfp
import collections
import numpy as np

LSTMState = collections.namedtuple('LSTMState', field_names=['h', 'c'])
ReadFilters = collections.namedtuple('ReadFilters', field_names=['F_X', 'F_Y', 'gamma'])
WriteFilters = collections.namedtuple('WriteFilters', field_names=['F_X', 'F_Y', 'gamma'])

class DRAW:
    def __init__(self, hps, name=None):
        self.img_height = hps.img_height
        self.img_width = hps.img_width
        self.img_channels = hps.img_channels
        self.read_dim = hps.read_dim
        self.write_dim = hps.write_dim
        self.enc_hidden_dim = hps.encoder_hidden_dim
        self.dec_hidden_dim = hps.decoder_hidden_dim
        self.z_dim = hps.z_dim
        self.num_timesteps = hps.num_timesteps
        self.init_scale = hps.init_scale
        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = name if name is not None else 'DRAW'

        with tf.variable_scope(self._name):

            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            self.do_inference = tf.placeholder(dtype=tf.bool, shape=[])
            batch_size = tf.shape(self.x)[0]

            init = tf.random_uniform_initializer(-self.init_scale, self.init_scale)
            tf.get_variable_scope().set_initializer(init)

            self.canvas_state_initial = self.get_initial_canvas_state(batch_size)
            self.enc_state_initial = self.get_initial_encoder_state(batch_size)
            self.dec_state_initial = self.get_initial_decoder_state(batch_size)

            self.canvas_array = tf.TensorArray(dtype=tf.float32, size=self.num_timesteps, infer_shape=True)
            self.dkl_z_array = tf.TensorArray(dtype=tf.float32, size=self.num_timesteps, infer_shape=True)

            loop_init_vars = (0, self.canvas_state_initial, self.enc_state_initial, self.dec_state_initial, self.dkl_z_array, self.canvas_array)
            loop_cond = lambda i, c, e, d, a1, a2: i < self.num_timesteps
            def loop_body(i, canvas, enc_state, dec_state, dkl_z_array, canvas_array):

                read_filters = self.compute_read_filters(dec_state.h)
                read_x = self.read_with_filters(self.x, read_filters)
                read_x_residual = self.read_with_filters(self.x-tf.nn.sigmoid(canvas), read_filters)
                enc_state_new = self.update_encoder_state(read_x, read_x_residual, enc_state)

                qz = self.qz(enc_state_new.h)
                pz = self.pz(batch_size)
                kl_div_z = qz.kl_divergence(pz)

                z = tf.where(self.do_inference, qz.sample(), pz.sample())
                
                dec_state_new = self.update_decoder_state(z, dec_state)

                w = self.brushstrokes(dec_state_new.h)
                write_filters = self.compute_write_filters(dec_state_new.h)
                canvas_new = self.write_with_filters(w, write_filters, canvas)

                return (i+1, canvas_new, enc_state_new, dec_state_new, dkl_z_array.write(i, kl_div_z), canvas_array.write(i, canvas_new))

            _, canvas_final, _, _, dkl_z_array_final, canvas_array_final = tf.while_loop(loop_cond, loop_body, loop_init_vars)

            self.dkl_Z = tf.reduce_sum(tf.transpose(dkl_z_array_final.stack()), axis=1)
            self.canvases = tf.transpose(canvas_array_final.stack(), perm=[1, 0, 2, 3, 4])
            self.drawings = tf.nn.sigmoid(self.canvases)

            self.log_prob_x_given_Z = self.px_given_canvas(canvas_final).log_prob(self.x)

            self.elbo_img = self.log_prob_x_given_Z - self.dkl_Z
            self.elbo = tf.reduce_mean(self.elbo_img, axis=0)
            self.loss = -self.elbo

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = [v for v in tf.trainable_variables() if v.name.startswith(self._name)]
            gradients, _ = zip(*self.optimizer.compute_gradients(loss=self.loss, var_list=tvars))
            self.train_op = self.optimizer.apply_gradients(grads_and_vars=zip(gradients, tvars), global_step=self.global_step)

            tf.summary.scalar('elbo', self.elbo)
            tf.summary.scalar('dkl_Z', tf.reduce_mean(self.dkl_Z, axis=0))
            self.merged_summaries = tf.summary.merge_all()

    def get_initial_canvas_state(self, batch_size):
        with tf.variable_scope('initial_canvas_state', reuse=tf.AUTO_REUSE):
            canvas = tf.get_variable(
                name='initial_canvas', dtype=tf.float32, shape=[self.img_height, self.img_width, self.img_channels],
                initializer=tf.zeros_initializer(), trainable=True)

            canvas = tf.tile(tf.expand_dims(canvas, 0), multiples=[batch_size, 1, 1, 1])
            return canvas

    def get_initial_encoder_state(self, batch_size):
        with tf.variable_scope('initial_encoder_state', reuse=tf.AUTO_REUSE):
            state = tf.get_variable(name='initial_encoder_state', dtype=tf.float32, shape=[2 * self.enc_hidden_dim],
                initializer=tf.zeros_initializer(), trainable=True)
            h, c = tf.split(state, 2, axis=0)
            h = tf.tile(tf.expand_dims(h, 0), multiples=[batch_size, 1])
            c = tf.tile(tf.expand_dims(c, 0), multiples=[batch_size, 1])
            return LSTMState(h=h, c=c)

    def get_initial_decoder_state(self, batch_size):
        with tf.variable_scope('initial_decoder_state', reuse=tf.AUTO_REUSE):
            state = tf.get_variable(name='initial_decoder_state', dtype=tf.float32, shape=[2 * self.dec_hidden_dim],
                initializer=tf.zeros_initializer(), trainable=True)
            h, c = tf.split(state, 2, axis=0)
            h = tf.tile(tf.expand_dims(h, 0), multiples=[batch_size, 1])
            c = tf.tile(tf.expand_dims(c, 0), multiples=[batch_size, 1])
            return LSTMState(h=h, c=c)

    def compute_read_filters(self, decoder_hidden_state):
         with tf.variable_scope('compute_read_filters', reuse=tf.AUTO_REUSE):
             five_numbers = tf.layers.dense(decoder_hidden_state, units=5, activation=None)
             g_tilde_X = five_numbers[:, 0]
             g_tilde_Y = five_numbers[:, 1]
             sigma_squared = tf.exp(five_numbers[:, 2])
             delta_tilde = tf.exp(five_numbers[:, 3])
             gamma = tf.exp(five_numbers[:, 4])
             F_X, F_Y = self.compute_filters(g_tilde_X, g_tilde_Y, sigma_squared, delta_tilde, gamma, self.read_dim)
             return ReadFilters(F_X=F_X, F_Y=F_Y, gamma=gamma)

    def compute_write_filters(self, decoder_hidden_state):
         with tf.variable_scope('compute_write_filters', reuse=tf.AUTO_REUSE):
             five_numbers = tf.layers.dense(decoder_hidden_state, units=5, activation=None)
             g_tilde_X = five_numbers[:, 0]
             g_tilde_Y = five_numbers[:, 1]
             sigma_squared = tf.exp(five_numbers[:, 2])
             delta_tilde = tf.exp(five_numbers[:, 3])
             gamma = tf.exp(five_numbers[:, 4])
             F_X, F_Y = self.compute_filters(g_tilde_X, g_tilde_Y, sigma_squared, delta_tilde, gamma, self.write_dim)
             return WriteFilters(F_X=F_X, F_Y=F_Y, gamma=gamma)

    def compute_filters(self, g_tilde_X, g_tilde_Y, sigma_squared, delta_tilde, gamma, N):
        g_X = tf.constant((float(self.img_width + 1) / 2.0)) * (g_tilde_X + 1.0) # [B]
        g_Y = tf.constant((float(self.img_height + 1) / 2.0)) * (g_tilde_Y + 1.0) # [B]
        delta = tf.constant((float(max(self.img_width, self.img_height) - 1.0) / float(N - 1))) * delta_tilde

        # a vector containing [1, 2, ..., N]
        window_ints = tf.cumsum(tf.ones(dtype=tf.int32, shape=[N]))
        window_ints = tf.cast(window_ints, dtype=tf.float32)
        centered_i_vec = window_ints - tf.constant((float(N) / 2.0)) - 0.5
        centered_j_vec = window_ints - tf.constant((float(N) / 2.0)) - 0.5

        # centered_i_vec and centered_j_vec give us some grids, delta scales the grid's overall size.
        # by adding these grid values elementwise to g_X and g_Y we obtain a set of NxN evenly spaced locations, 
        # centered at g_X, g_Y and with total width controlled by delta.
        # each of these locations will be the mean of a 2D gaussian kernel.

        # the gaussian kernels weight the pixels in the image, in a weighted sum. 
        # in this manner, we obtain a soft attention window from our parametrizable grid of gaussian kernels.

        mu_X_vec = tf.expand_dims(g_X, 1) + tf.expand_dims(centered_j_vec, 0) * tf.expand_dims(delta, 1) # [B, N]
        mu_Y_vec = tf.expand_dims(g_Y, 1) + tf.expand_dims(centered_i_vec, 0) * tf.expand_dims(delta, 1) # [B, N]
        mu_X_vec = tf.expand_dims(mu_X_vec, 2) # [B, N, 1]
        mu_Y_vec = tf.expand_dims(mu_Y_vec, 2) # [B, N, 1]

        img_position_ints_X = tf.cumsum(tf.ones(dtype=tf.int32, shape=(self.img_width))) - 1
        img_position_ints_Y = tf.cumsum(tf.ones(dtype=tf.int32, shape=(self.img_height))) - 1
        img_position_ints_X = tf.cast(img_position_ints_X, dtype=tf.float32)
        img_position_ints_Y = tf.cast(img_position_ints_Y, dtype=tf.float32)
        img_position_ints_X = tf.expand_dims(tf.expand_dims(img_position_ints_X, 0), 1) # [1, 1, W]
        img_position_ints_Y = tf.expand_dims(tf.expand_dims(img_position_ints_Y, 0), 1) # [1, 1, H]
        
        F_X_exp_arg_numerators = tf.square(img_position_ints_X - mu_X_vec) # [B, N, W]
        F_Y_exp_arg_numerators = tf.square(img_position_ints_Y - mu_Y_vec) # [B, N, H]
        F_X_exp_arg_denominators = 2.0 * tf.expand_dims(tf.expand_dims(sigma_squared, -1), -1) # [B, 1, 1]
        F_Y_exp_arg_denominators = 2.0 * tf.expand_dims(tf.expand_dims(sigma_squared, -1), -1) # [B, 1, 1]

        F_X_exp_args = -(F_X_exp_arg_numerators / F_X_exp_arg_denominators) # [B, N, W]
        F_Y_exp_args = -(F_Y_exp_arg_numerators / F_Y_exp_arg_denominators) # [B, N, H]
        
        F_X_exps = tf.exp(F_X_exp_args)
        F_Y_exps = tf.exp(F_Y_exp_args)

        # normalizing constants
        Z_X = tf.maximum(1e-8, tf.reduce_sum(F_X_exps, axis=2, keep_dims=True))
        Z_Y = tf.maximum(1e-8, tf.reduce_sum(F_Y_exps, axis=2, keep_dims=True))

        F_X = (F_X_exps / Z_X)  # [B, N, W]
        F_Y = (F_Y_exps / Z_Y)  # [B, N, H]

        return F_X, F_Y

    def read_with_filters(self, x, read_filters):
        F_X = read_filters.F_X
        F_Y = read_filters.F_Y
        gamma = read_filters.gamma
        read_x = tf.einsum('bmh,bhnc->bmnc', F_Y, tf.einsum('bhwc,bnw->bhnc', x, F_X))
        read_x = tf.einsum('b,bmnc->bmnc', gamma, read_x)
        return read_x

    def write_with_filters(self, w, write_filters, canvas):
        F_X = write_filters.F_X
        F_Y = write_filters.F_Y
        gamma = write_filters.gamma
        written_w = tf.einsum('bmh,bmwc->bhwc', F_Y, tf.einsum('bmnc,bnw->bmwc', w, F_X))
        written_w = tf.einsum('b,bhwc->bhwc', (1.0 / (1e-8 + gamma)), written_w)
        canvas_new = canvas + written_w
        return canvas_new

    def brushstrokes(self, decoder_hidden_state):
        with tf.variable_scope('brushstrokes', reuse=tf.AUTO_REUSE):
            w_flat = tf.layers.dense(decoder_hidden_state, units=(self.write_dim * self.write_dim * self.img_channels), 
                activation=None)
            w = tf.reshape(w_flat, [-1, self.write_dim, self.write_dim, self.img_channels])
            return w

    def update_encoder_state(self, r_x, r_x_residual, enc_state_prev):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            h_prev, c_prev = enc_state_prev.h, enc_state_prev.c

            vec = tf.concat([tf.layers.flatten(r_x), tf.layers.flatten(r_x_residual), h_prev], axis=-1)
            fioj = tf.layers.dense(vec, units=(4 * self.enc_hidden_dim), activation=None)

            f, i, o, j = tf.split(fioj, 4, axis=1)
            f = tf.nn.sigmoid(f+1.0)
            i = tf.nn.sigmoid(i)
            o = tf.nn.sigmoid(o)
            j = tf.nn.tanh(j)

            c = f * c_prev + i * j
            h = o * c
            return LSTMState(h=h, c=c)

    def update_decoder_state(self, z, dec_state_prev):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            h_prev, c_prev = dec_state_prev.h, dec_state_prev.c

            vec = tf.concat([z, h_prev], axis=1)
            fioj = tf.layers.dense(vec, units=(4 * self.dec_hidden_dim), activation=None)
            
            f, i, o, j = tf.split(fioj, 4, axis=1)
            f = tf.nn.sigmoid(f+1.0)
            i = tf.nn.sigmoid(i)
            o = tf.nn.sigmoid(o)
            j = tf.nn.tanh(j)

            c = f * c_prev + i * j
            h = o * c
            return LSTMState(h=h, c=c)

    def qz(self, enc_hidden_state):
        with tf.variable_scope('qz', reuse=tf.AUTO_REUSE):
            fc = tf.layers.dense(enc_hidden_state, units=(2 * self.z_dim), activation=None)
            mu, logsigma = tf.split(fc, 2, axis=1)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

    def pz(self, batch_size):
        with tf.variable_scope('pz', reuse=tf.AUTO_REUSE):
            mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.z_dim])
            logsigma = tf.zeros(dtype=tf.float32, shape=[batch_size, self.z_dim])
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

    def px_given_canvas(self, canvas):
        with tf.variable_scope('px_given_canvas', reuse=tf.AUTO_REUSE):
            x_dist = tfp.distributions.Bernoulli(logits=canvas)
            x_dist = tfp.distributions.Independent(x_dist)
            return x_dist

    def train(self, sess, x):
        feed_dict = {
            self.x: x,
            self.do_inference: True
        }
        _, elbo, step, summaries = sess.run([self.train_op, self.elbo, self.global_step, self.merged_summaries], feed_dict=feed_dict)
        return elbo, step, summaries

    def reconstruct(self, sess, x):
        feed_dict = {
            self.x: x,
            self.do_inference: True
        }
        drawings = sess.run(self.drawings, feed_dict=feed_dict)
        return drawings

    def generate(self, sess, num_samples):
        batch_size = num_samples
        x = np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width, self.img_channels))
        feed_dict = {
            self.x: x,
            self.do_inference: False
        }
        drawings = sess.run(self.drawings, feed_dict=feed_dict)
        return drawings
