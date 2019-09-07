import tensorflow as tf
import tensorflow_datasets as tfds

SUPPORTED_DATASETS = ['mnist', 'celeb_a', 'cifar10', 'omniglot', 'svhn_cropped']

def get_dataset(name, hps):
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise NotImplementedError('Dataset not supported.')

    ds_train, ds_test = tfds.load(name=name, split=["train", "test"])

    ds_train = preprocess_dataset(name, ds_train, hps, epochs=hps.epochs)
    ds_test = preprocess_dataset(name, ds_test, hps, epochs=1)
    return ds_train, ds_test

def preprocess_dataset(name, dataset, hps, epochs):
    img_height = hps.img_height
    img_width = hps.img_width
    batch_size = hps.batch_size
    
    f1 = lambda row: row["image"]
    f2 = lambda img: tf.cast(img, dtype=tf.int32)
    f3 = lambda img: tf.cast(img, dtype=tf.float32)

    f4 = lambda img: img * tf.constant((1.0 / 256.0))

    if name == 'celeb_a':
        f5a = lambda img: img[((tf.shape(img)[0]//2)-54):((tf.shape(img)[0]//2)+54), ((tf.shape(img)[1]//2)-54):((tf.shape(img)[1]//2)+54)]
        f5b = lambda img: tf.image.resize_images(img, [hps.img_height, hps.img_width])
        f5 = lambda img: f5b(f5a(img))
    else:
        f5 = lambda img: tf.image.resize_images(img, [hps.img_height, hps.img_width])

    f6 = lambda img: img + tf.random_uniform(minval=0.0, maxval=(1.0/256.0), shape=[hps.img_height, hps.img_width, hps.img_channels])
    f7 = lambda img: tf.cast(img, dtype=tf.float32)

    normalize_pixels = lambda row: f7(f6(f5(f4(f3(f2(f1(row)))))))
    ds = dataset.map(normalize_pixels)
    ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.repeat(epochs)

    return ds

