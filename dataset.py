import tensorflow as tf
import numpy as np
STDDEV = 1.
BATCH_SIZE = 32


def normalize_images(images):
    images = (images.astype('float32') - 127.5) / 127.5
    images = np.pad(images, [(0, 0), (2, 2), (2, 2), (0, 0)], 'constant', constant_values=(-1))
    return images


def get_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis], normalize_images(x_test[..., np.newaxis])
    x_train, x_val = normalize_images(x_train[:50000]), normalize_images(x_train[50000:])
    return x_train, x_val, x_test


def add_noise(next_elem, loc=0, scale=STDDEV, normalize=False):
    noise = tf.random.normal(shape=next_elem.shape, mean=loc, stddev=scale)
    noised = next_elem + noise
    if normalize:
        noised = (noised - np.min(noised)) / np.ptp(noised)
    return noised


def process_image(img, std=STDDEV):
    img_noise = add_noise(img, scale=std)
    return img_noise, img


def get_datasets():
    sets = get_data()
    sets = [tf.data.Dataset.from_tensor_slices(s) for s in sets]
    train_dataset, val_dataset, test_dataset = [s.map(process_image) for s in sets]
    return train_dataset.batch(BATCH_SIZE), val_dataset.batch(BATCH_SIZE), test_dataset.batch(BATCH_SIZE)