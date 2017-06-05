import gzip, numpy, pickle


def convert_number_to_vector(number_array):
    n = number_array.shape[0]
    vector_of_number = numpy.zeros((n, 10))

    for i in range(0, n):
        vector_of_number[i][number_array[i]] = 1

    return vector_of_number


def get_inputs(path):
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    X_train_set = train_set[0]
    Y_train_set_n = train_set[1]
    X_valid_set = valid_set[0]
    Y_valid_set_n = valid_set[1]
    X_test_set = test_set[0]
    Y_test_set_n = test_set[1]

    Y_train_set_v = convert_number_to_vector(Y_train_set_n)
    Y_valid_set_v = convert_number_to_vector(Y_valid_set_n)
    Y_test_set_v = convert_number_to_vector(Y_test_set_n)

    return X_train_set, Y_train_set_n, X_valid_set, Y_valid_set_n, X_test_set, Y_test_set_n, \
        Y_train_set_v, Y_valid_set_v, Y_test_set_v
