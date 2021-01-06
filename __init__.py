import pathlib
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(path, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size[0], image_size[1]])
    return image


def show_img(train_dataset):
    import matplotlib.pyplot as plt
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')


def custom(class_names, data_root_orig, batch, image_size):
    data_root = pathlib.Path(data_root_orig)
    all_image_paths = list(data_root.rglob('*/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths if pathlib.Path(path).is_file()]

    image_count = len(all_image_paths)

    label_names = []
    for path in all_image_paths:
        if pathlib.Path(path).parent.name == class_names[0]:
            label_names.append(0)
        else:
            label_names.append(1)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(lambda path: preprocess_image(path, image_size), num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label_names, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # Установка размера буфера перемешивания, равного набору данных, гарантирует
    # полное перемешивание данных.
    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(batch)

    size_ds = image_count
    train_size = int(0.8*size_ds)
    test_size = int(0.2*size_ds)
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    test_ds = ds.take(test_size)

    return train_ds, test_ds
