import matplotlib.pyplot as pltimport numpy as npimport collectionsimport csvfrom tensorflow.python.framework import dtypesfrom tensorflow.python.platform import gfileDataset = collections.namedtuple('Dataset', ['data', 'target'])Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])VIDEO_TRAINING = "/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/train_data/training_16by16_fewer.csv"# VIDEO_TRAINING = "/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/training_data_4_fake_without_comma.csv"VIDEO_TESTING = "/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/test_data/testing_16by16_35.csv"def dense_to_one_hot(labels_dense, num_classes):    """Convert class labels from scalars to one-hot vectors."""    num_labels = labels_dense.shape[0]    print("number of labels : " + str(num_labels))    index_offset = np.arange(num_labels) * num_classes    labels_one_hot = np.zeros((num_labels, num_classes))    print("type of labels_one_hot :     " + str(type(labels_one_hot)))    print("shape of labels_one_hot :     " + str(labels_one_hot.shape))    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1    return labels_one_hot    # num_labels = labels_dense.shape[0]    # index_offset = numpy.arange(num_labels) * num_classes    # labels_one_hot = numpy.zeros((num_labels, num_classes))    # labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1    # return labels_one_hotdef load_csv_without_header(filename,                            target_dtype,                            features_dtype,                            n_samples,                            block_size,                            target_column=-1,                            ):    """Load dataset from CSV file with a header row."""    with gfile.Open(filename) as csv_file:        data_file = csv.reader(csv_file)        # header = next(data_file)        n_samples = n_samples        # n_samples = 5124        n_features = block_size ** 2        data = np.zeros((n_samples, n_features), dtype=features_dtype)        target = np.zeros((n_samples,), dtype=target_dtype)        for i, row in enumerate(data_file):            qwer = np.asarray(row.pop(0), dtype=target_dtype)            target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)            data[i] = np.asarray(row, dtype=features_dtype)    # print(type(data))    # print(data)    # print(data.ndim)    # print("==============================")    # print("============================== now flatten")    data = data.flatten()    data = data.reshape(n_samples, block_size, block_size, 1)    # target = target.flatten()    # target = target.reshape(n_samples, block_size, block_size, 1)    # target = dense_to_one_hot(target, 37)    # print(type(data))    # print(data)    # print(len(data))    # print(data.ndim)    return Dataset(data=data, target=target)class DataSet(object):    def __init__(self,                 images,                 labels,                 fake_data=False,                 one_hot=False,                 dtype=dtypes.float32,                 reshape=True):        """Construct a DataSet.        one_hot arg is used only if fake_data is true.  `dtype` can be either        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into        `[0, 1]`.        """        dtype = dtypes.as_dtype(dtype).base_dtype        if dtype not in (dtypes.uint8, dtypes.float32):            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %                            dtype)        if fake_data:            self._num_examples = 10000            self.one_hot = one_hot        else:            assert images.shape[0] == labels.shape[0], (                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))            self._num_examples = images.shape[0]            # Convert shape from [num examples, rows, columns, depth]            # to [num examples, rows*columns] (assuming depth == 1)            if reshape:                assert images.shape[3] == 1                images = images.reshape(images.shape[0],                                        images.shape[1] * images.shape[2])            if dtype == dtypes.float32:                # Convert from [0, 255] -> [0.0, 1.0].                images = images.astype(np.float32)                images = np.multiply(images, 1.0 / 255.0)        self._images = images        self._labels = labels        self._epochs_completed = 0        self._index_in_epoch = 0    @property    def images(self):        return self._images    @property    def labels(self):        return self._labels    @property    def num_examples(self):        return self._num_examples    @property    def epochs_completed(self):        return self._epochs_completed    def next_batch(self, batch_size, fake_data=False):        """Return the next `batch_size` examples from this data set."""        start = self._index_in_epoch        self._index_in_epoch += batch_size        if self._index_in_epoch > self._num_examples:            # Finished epoch            self._epochs_completed += 1            # Shuffle the data            perm = np.arange(self._num_examples)            np.random.shuffle(perm)            self._images = self._images[perm]            self._labels = self._labels[perm]            # Start next epoch            start = 0            self._index_in_epoch = batch_size            assert batch_size <= self._num_examples        end = self._index_in_epoch        return self._images[start:end], self._labels[start:end]def read_data_sets(dtype=dtypes.float32,                   reshape=True,                   validation_size=0):    # r = csv.reader(open(VIDEO_TRAINING))  # Here your csv file    # row_count = sum(1 for row in r)    # print("training data has " + str(row_count) + " rows")    # training set start-------------------------------------------------------->    # train_set = load_csv_without_header(filename=VIDEO_TRAINING, target_dtype=np.int, features_dtype=np.int,    #                                     n_samples=row_count, block_size=16)  # 4328116    # training set end -------------------------------------------------------->    # n_samples=51, block_size=4)  # 4328116    # train_images = train_set.data    # train_labels = train_set.target    r = csv.reader(open(VIDEO_TESTING))  # Here your csv file    row_count = sum(1 for row in r)    print("Testing data has " + str(row_count) + " rows")    # testing set start -------------------------------------------------------->    test_set = load_csv_without_header(        filename=VIDEO_TESTING,        target_dtype=np.int,        features_dtype=np.int,        n_samples=row_count,        block_size=16,    )    # testing set end -------------------------------------------------------->    test_images = test_set.data    test_labels = test_set.target    # if not 0 <= validation_size <= len(train_images):    #     raise ValueError(    #         'Validation size should be between 0 and {}. Received: {}.'    #             .format(len(train_images), validation_size))    # validation_images = train_images[:validation_size]    # validation_labels = train_labels[:validation_size]    # train_images = train_images[validation_size:]    # train_labels = train_labels[validation_size:]    # train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)    # validation = DataSet(validation_images,    #                      validation_labels,    #                      dtype=dtype,    #                      reshape=reshape)    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)    # return Datasets(train=train, validation=validation, test=test)    return testnp.set_printoptions(threshold=10000000000000)test = read_data_sets(reshape=False, validation_size=0)print("============")print("test target is : ")print(test.labels[1])print(test.labels[1].shape)# print(test.images[0:1])print(test.images[1].shape)x = test.images[18:19].flatten()# print('the size after flatten is: ')# print(x.shape)x_img = x.reshape(16, 16)print(x_img)print(x_img.shape)# plt.imshow(x_img, cmap='gray')plt.imshow(x_img)plt.show()# print(type(test.images))# print(test.images.size)# print(test.images.shape)