import numpy as np

# all example
def normalize_coordinate(vertices):
    mean = np.mean(vertices, 1)
    mean = np.reshape(mean, (-1, 1, vertices.shape[2]))
    vertices = vertices - mean
    max = np.max(np.absolute(vertices), 1)
    max = np.reshape(max, (-1, 1, vertices.shape[2]))
    vertices = np.divide(vertices, max, out=np.zeros_like(vertices), where=max!=0)
    return vertices

######################
# Each example at once
######################

def angle_between_faces(i, face_adjacency_angles):
    return face_adjacency_angles[i]

def get_vertex(edge, face):
    for i in range(len(face)):
        if (face[i] not in edge):
            return i 

def get_opposite_angles(i, edges, face_adjacency, faces, face_angles):
    two_angle = []
    for face_index in face_adjacency[i]:
        vertex_index = get_vertex(edges[i], faces[face_index])
        two_angle.append(face_angles[face_index][vertex_index])
    return two_angle

def d(p, q, r):
    d = (q - p) / np.linalg.norm(q - p)
    v = r - p
    t = np.dot(v, d)
    s = p +  t * d
    return np.linalg.norm(s-r)

def get_two_ratios(i, edges, vertices, face_adjacency, faces):
    two_ratio = []
    for face_index in face_adjacency[i]:
        vertex_index = get_vertex(edges[i], faces[face_index])
        r = vertices[faces[face_index][vertex_index]]
        p = vertices[edges[i][0]]
        q = vertices[edges[i][1]]
        edge_length = np.linalg.norm(p-q)
        distance = d(p, q, r)
        ratio = distance/ edge_length
        two_ratio.append(ratio)
    return two_ratio

def extract_features(features, NUM_EDGES):

    edges = features['edges']
    vertices = features['vertices']
    faces = features['faces']
    face_adjacency_angles = features['face_adjacency_angles']
    face_angles = features['face_angles']
    face_adjacency = features['face_adjacency']
    
    normalize_coordinate(vertices)
    mesh_features = []

    for i in range(features['edges'].shape[0]):
        features_per_obj = []
        for j in range(NUM_EDGES):
            feature = []
            feature.append(angle_between_faces(j, face_adjacency_angles[i]))
            feature.extend(get_opposite_angles(j, edges[i], face_adjacency[i], faces[i], face_angles[i]))
            feature.extend(get_two_ratios(j, edges[i], vertices[i], face_adjacency[i], faces[i]))
            features_per_obj.append(feature)
        
        mesh_features.append(features_per_obj)
    
    mesh_features = np.array(mesh_features)
    return mesh_features