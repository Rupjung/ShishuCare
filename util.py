import librosa
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


# ---------------- Attention Layer ----------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


# ---------------- Model & Encoder Loader ----------------
def load_model_and_encoder():
    model = tf.keras.models.load_model(
        "baby_cry_classifier.h5",
        compile=False,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    le = joblib.load("label_encoder_1.pkl")   # make sure this path is correct
    return model, le


# ---------------- Feature Extractor ----------------
def extract_features(file_path, max_len=216):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape (128, time)

    # Pad or truncate to fixed length
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]

    # Add channel + batch dims → (1, 128, 216, 1)
    mel_db = np.expand_dims(mel_db, axis=-1)
    return np.expand_dims(mel_db, axis=0)


# ---------------- Prediction ----------------
def predict_audio(file_path):
    model, le = load_model_and_encoder()
    features = extract_features(file_path)
    prediction = model.predict(features)[0]          # probabilities
    class_idx = np.argmax(prediction)               # predicted index
    label = le.inverse_transform([class_idx])[0]    # ✅ fixed bug
    confidence = float(prediction[class_idx])
    return label, confidence