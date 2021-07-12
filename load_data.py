from data_preprocess.extract_features import extract_features
from data_preprocess.data_reader import tfrecord_to_dataset

NUM_EDGES = 750
if __name__ == "__main__":
    features, labels = tfrecord_to_dataset('./dataset/encoded_data/train.tfrecord')
    mesh_features = extract_features(features, NUM_EDGES)
    print(mesh_features.shape)