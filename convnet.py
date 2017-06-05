import tensorflow as tf
import inputs
import numpy as np


class ConvNet:
    def __init__(self, w, z):
        self.X_train_set, self.Y_train_set_n, self.X_valid_set, self.Y_valid_set_n, self.X_test_set,\
            self.Y_test_set_n, self.Y_train_set_v, self.Y_valid_set_v, self.Y_test_set_v =\
            inputs.get_inputs('mnist.pkl.gz')

        self.n_training_examples = self.X_train_set.shape[0]

        self.max_epochs = w
        self.batch_size = z
        self.w = {}

        self.w['conv_1'] = {'weights': tf.Variable(tf.random_normal((5, 5, 1, 32))),
                            'biases': tf.Variable(tf.random_normal((1, 32)))}

        self.w['conv_2'] = {'weights': tf.Variable(tf.random_normal((5, 5, 32, 64))),
                            'biases': tf.Variable(tf.random_normal((1, 64)))}

        self.w['fc_1'] = {'weights': tf.Variable(tf.random_normal((7*7*64, 1024))),
                          'biases': tf.Variable(tf.random_normal((1, 1024)))}

        self.w['fc_2'] = {'weights': tf.Variable(tf.random_normal((1024, 10))),
                          'biases': tf.Variable(tf.random_normal((1, 10)))}

    def predict(self, X):        

        self.l0 = X

        self.l1 = tf.nn.conv2d(X, self.w['conv_1']['weights'], strides=(1, 1, 1, 1), padding='SAME')
        self.l1 = tf.add(self.w['conv_1']['biases'], self.l1)
        self.l1 = tf.nn.relu(self.l1)

        self.l1 = tf.nn.max_pool(self.l1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

        self.l2 = tf.nn.conv2d(self.l1, self.w['conv_2']['weights'], strides=(1, 1, 1, 1), padding='SAME')
        self.l2 = tf.add(self.l2, self.w['conv_2']['biases'])
        self.l2 = tf.nn.relu(self.l2)

        self.l2 = tf.nn.max_pool(self.l2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        self.l2 = tf.reshape(self.l2, (-1, 7*7*64))

        self.l3 = tf.matmul(self.l2, self.w['fc_1']['weights'])
        self.l3 = tf.add(self.l3, self.w['fc_1']['biases'])
        self.l3 = tf.nn.relu(self.l3)

        self.l4 = tf.matmul(self.l3, self.w['fc_2']['weights'])
        self.l4 = tf.add(self.l4, self.w['fc_2']['biases'])

        return self.l4

    def train(self):

        X = tf.placeholder(dtype='float', shape=(None, 784))

        res = tf.reshape(X, shape=[-1, 28, 28, 1])

        Y = tf.placeholder(dtype='float', shape=(None, 10))

        predicted_vector = self.predict(res)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_vector, labels=Y))

        optimiser = tf.train.AdamOptimizer().minimize(loss)
        n_updates_per_epoch = int(self.n_training_examples / self.batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
  
            for i in range(self.max_epochs):
                epoch_loss = 0
                for j in range(n_updates_per_epoch):

                    current_batch_X = self.X_train_set[j * self.batch_size: j * self.batch_size + self.batch_size]

                    current_batch_Y = self.Y_train_set_v[j * self.batch_size: j * self.batch_size + self.batch_size]

                    _, temp = sess.run((optimiser, loss), feed_dict={X: current_batch_X, Y: current_batch_Y})
                    epoch_loss += temp
                print(str(i) + ' epoch loss is ' + str(epoch_loss))

                predi = sess.run(self.predict(np.reshape(self.X_valid_set, (-1, 28, 28, 1))))
                prediction = np.argmax(predi, axis=1)
                correct = self.Y_valid_set_n
                accuracy = np.mean(np.equal(prediction, correct).astype(float))
                g = str('epoch ' + str(i) + ' accuracy is ' + str(accuracy * 100) + ' epoch_loss ' + str(epoch_loss))
                print(g)

cn1 = ConvNet(10, 100)
cn1.train()
