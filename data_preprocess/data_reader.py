import trimesh
import numpy as np
import os
import tensorflow as tf

# Create a description of the features.
feature_description = {
    'num_vertices': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'num_faces': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'vertices': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'faces': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'edges': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'face_adjacency_angles': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'face_angles': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'face_adjacency': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'labels': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  example = tf.io.parse_single_example(example_proto, feature_description)
  features = dict(
    num_vertices = example['num_vertices'],
    num_faces = example['num_faces'],
    vertices = tf.io.parse_tensor(example['vertices'], out_type = tf.float32),
    faces = tf.io.parse_tensor(example['faces'], out_type = tf.int32),
    edges = tf.io.parse_tensor(example['edges'], out_type = tf.int32),
    face_adjacency_angles = tf.io.parse_tensor(example['face_adjacency_angles'], out_type = tf.float32),
    face_angles = tf.io.parse_tensor(example['face_angles'], out_type = tf.float32),
    face_adjacency = tf.io.parse_tensor(example['face_adjacency'], out_type = tf.int32),
    labels = example['labels']
  )
  return features

def tfrecord_to_dataset(data_path):
    raw_dataset = tf.data.TFRecordDataset(data_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    features = dict(
        num_vertices = [],
        num_faces = [],
        vertices = [],
        faces = [],
        edges = [],
        face_adjacency_angles = [],
        face_angles = [],
        face_adjacency = [],
    )
    labels = []
    for data in parsed_dataset:
        num_vertices = np.asarray(data['num_vertices']).item()
        num_faces = np.asarray(data['num_vertices']).item()
        vertices = np.asarray(data['vertices'])
        faces = np.asarray(data['faces'])
        edges = np.asarray(data['edges'])
        face_adjacency_angles = np.asarray(data['face_adjacency_angles'])
        face_angles = np.asarray(data['face_angles'])
        face_adjacency = np.asarray(data['face_adjacency'])
        faces = np.asarray(data['faces'])
        label = np.asarray(data['labels']).item()

        features['num_vertices'].append(num_vertices)
        features['num_faces'].append(num_faces)
        features['vertices'].append(vertices)
        features['faces'].append(faces)
        features['edges'].append(edges)
        features['face_adjacency_angles'].append(face_adjacency_angles)
        features['face_angles'].append(face_angles)
        features['face_adjacency'].append(face_adjacency)
        labels.append(label)

    features['num_vertices'] = np.array(features['num_vertices'])
    features['num_faces'] = np.array(features['num_faces'])
    features['vertices'] = np.array(features['vertices'])
    features['faces'] = np.array(features['faces'])
    features['edges'] = np.array(features['edges'])
    features['face_adjacency_angles'] = np.array(features['face_adjacency_angles'])
    features['face_angles'] = np.array(features['face_angles'])
    features['face_adjacency'] = np.array(features['face_adjacency'])
    labels = np.array(labels)

    return features, labels

def _tensor_feature(values, dtype):
    values = tf.dtypes.cast(values, dtype)
    serialised_values = tf.io.serialize_tensor(values)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialised_values.numpy()]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def dataset_to_tfrecord(dir_path, tfrecord_file, classes):
    """
    dir_path: directory of data, contains subdirectories corresponding to classes
    tfrecord_file: tfrecord file to write data into
    classes: dictionary of classes name and index
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for subdir_name in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir_name)
            for filename in os.listdir(subdir_path):
                path = os.path.join(subdir_path, filename)
                # load 3d object
                mesh_data = trimesh.load_mesh(path, "stl")
                num_vertices = mesh_data.vertices.shape[0]
                num_faces = mesh_data.faces.shape[0]
                vertices = np.asarray(mesh_data.vertices)
                edges = np.asarray(mesh_data.edges_unique)
                faces = np.asarray(mesh_data.faces)
                # angles between 2 faces share by edge in order of edges_unique
                face_adjacency_angles = np.asarray(mesh_data.face_adjacency_angles)
                # angle at each vertex of face
                face_angles = np.asarray(mesh_data.face_angles)
                # faces share by edge corresponding to order in edges_unique
                face_adjacency = np.asarray(mesh_data.face_adjacency)
                labels = classes[subdir_name]
                
                feature = {
                            'num_vertices': _int64_feature(num_vertices),
                            'num_faces': _int64_feature(num_faces),
                            'vertices': _tensor_feature(vertices, tf.float32),
                            'faces': _tensor_feature(faces, tf.int32),
                            'edges': _tensor_feature(edges, tf.int32),
                            'face_adjacency_angles': _tensor_feature(face_adjacency_angles, tf.float32),
                            'face_angles': _tensor_feature(face_angles, tf.float32),
                            'face_adjacency': _tensor_feature(face_adjacency, tf.int32),
                            'labels': _int64_feature(labels)
                            }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized_proto = example_proto.SerializeToString()
                
                writer.write(serialized_proto)

def mesh_to_dataset(file_path):
       
    mesh_data = trimesh.load_mesh(file_path, "stl")
    num_vertices = mesh_data.vertices.shape[0]
    num_faces = mesh_data.faces.shape[0]
    vertices = np.asarray(mesh_data.vertices)
    edges = np.asarray(mesh_data.edges_unique)
    faces = np.asarray(mesh_data.faces)
    # angles between 2 faces share by edge in order of edges_unique
    face_adjacency_angles = np.asarray(mesh_data.face_adjacency_angles)
    # angle at each vertex of face
    face_angles = np.asarray(mesh_data.face_angles)
    # faces share by edge corresponding to order in edges_unique
    face_adjacency = np.asarray(mesh_data.face_adjacency)
    features = dict(
        num_vertices = [],
        num_faces = [],
        vertices = [],
        faces = [],
        edges = [],
        face_adjacency_angles = [],
        face_angles = [],
        face_adjacency = [],
    )
    features['num_vertices'].append(num_vertices)
    features['num_faces'].append(num_faces)
    features['vertices'].append(vertices)
    features['faces'].append(faces)
    features['edges'].append(edges)
    features['face_adjacency_angles'].append(face_adjacency_angles)
    features['face_angles'].append(face_angles)
    features['face_adjacency'].append(face_adjacency)

    features['num_vertices'] = np.array(features['num_vertices'])
    features['num_faces'] = np.array(features['num_faces'])
    features['vertices'] = np.array(features['vertices'])
    features['faces'] = np.array(features['faces'])
    features['edges'] = np.array(features['edges'])
    features['face_adjacency_angles'] = np.array(features['face_adjacency_angles'])
    features['face_angles'] = np.array(features['face_angles'])
    features['face_adjacency'] = np.array(features['face_adjacency'])

    return features
    
    