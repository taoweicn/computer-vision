import tensorflow as tf
import argparse
from get_data import get_data, get_data_random_crop
from models import fcn_8s
import matplotlib.pyplot as plt
import pandas as pd


keras = tf.keras
VGG16 = keras.applications.vgg16.VGG16
vgg16 = keras.applications.vgg16
K = keras.backend
AUTOTUNE = tf.data.experimental.AUTOTUNE

# tf.enable_eager_execution()


def test_model_without_val():

    data, n = get_data()
    # data = data.repeat(1000).batch(1).prefetch(AUTOTUNE)
    data = data.batch(2).prefetch(AUTOTUNE)
    callbacks = [keras.callbacks.TensorBoard('./logs/not_val/new')]
    model = fcn_8s()
    # print(model.layers[2].get_config())
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # 1e-5?
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    history = model.fit(data, epochs=100, callbacks=callbacks)
    # history = model.fit(data, epochs=250)
    model.save('./models/no_val/first_model_train3.h5')
    # pred = model.predict(data.map(get_x, AUTOTUNE))
    # pred = np.argmax(pred, axis=-1)

    pass

    # x = np.random.random((7, 512, 256, 3)).astype(np.float32)


def test_model_random_crop():
    data, n = get_data_random_crop(10)
    data = data.batch(2).prefetch(AUTOTUNE)
    model = fcn_8s()

    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # 1e-5?
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    history = model.fit(data, epochs=10, steps_per_epoch=50, shuffle=True)

    def plot_learning_curves(history):
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 3)
        plt.show()

    plot_learning_curves(history)


def test_model():
    callbacks = [keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                 keras.callbacks.TensorBoard('./logs/val')]
    model = fcn_8s()
    model.compile(optimizer=keras.optimizers.Adam(1e-6), loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    data, n = get_data()
    # repeat before batch for better performance
    train_data = data.take(int(0.8 * n)).repeat(100).batch(1).prefetch(AUTOTUNE)
    val_data = data.skip(int(0.8 * n)).batch(1).prefetch(AUTOTUNE)

    # train_data, val_data, n = get_data_v5()
    # train_data = train_data.batch(1).repeat(100)
    # val_data = val_data.batch(1)

    history = model.fit(train_data, epochs=10000, steps_per_epoch=20, validation_data=val_data, callbacks=callbacks)
    model.save('./models/val/first_model_t2.h5')


if __name__ == '__main__':
    test_model_random_crop()

    # parser = argparse.ArgumentParser(description='test fcn_8s')
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('--random_crop', '-c', action='store_true', help='with random crop')
    # group.add_argument('--with_val', '-v', action='store_true', help='with validation')
    # args = parser.parse_args()
    #
    # if args.random_crop:
    #     test_model_random_crop()
    # elif args.with_val:
    #     test_model()
    # else:
    #     test_model_without_val()
