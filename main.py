from dataset import get_datasets
from model import create_model
from tensorflow.keras.optimizers import Adam


def train():
    train_dataset, _, _ = get_datasets()
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=4e-1), loss='mse')
    model.fit(train_dataset, epochs=5)
    return model


if __name__ == '__main__':
    model = train()
    model.save('saved_model')
