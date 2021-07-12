from load_data import NUM_EDGES
from data_preprocess.extract_features import extract_features
from data_preprocess.data_reader import mesh_to_dataset

NUM_EDGES = 750

if __name__ == "__main__":
    features = mesh_to_dataset('./dataset/clean_data/train/cube/1.stl')
    mesh_features = extract_features(features, NUM_EDGES)
    print(mesh_features.shape)