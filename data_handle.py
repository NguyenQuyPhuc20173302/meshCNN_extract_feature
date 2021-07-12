from data_preprocess.data_reader import tfrecord_to_dataset, dataset_to_tfrecord

CLASSES = {'cube': 0, 'sphere': 1}
if __name__ == "__main__":
    train_data_path = '.\\dataset\\clean_data\\train'
    test_data_path = '.\\dataset\\clean_data\\test'
    train_tfrecord_file = './dataset/encoded_data/train.tfrecord'
    test_tfrecord_file = './dataset/encoded_data/test.tfrecord'
    dataset_to_tfrecord(train_data_path, train_tfrecord_file, CLASSES)
    dataset_to_tfrecord(test_data_path, test_tfrecord_file, CLASSES)

    

