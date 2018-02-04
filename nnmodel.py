import tensorflow as tf
import pandas as pd
import numpy as np


class model:
    def __init__(self, actions, statenum):
        self.actions = actions
        self.statenum = statenum
        self.sess = tf.Session()
        self.x = tf.placeholder(shape=[None, 4], dtype=np.float32)
        self.y = tf.placeholder(shape=[None, 4], dtype=np.float32)

        self.w1 = tf.Variable(tf.ones(shape=[4, 4]), dtype=np.float32)
        self.logits = tf.nn.relu(tf.matmul(self.x, self.w1))
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.logits)))
        self.optimizer = tf.train.GradientDescentOptimizer(1).minimize(self.loss)
        self.initiation = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.timer = 0
        self.ckptdir = '/home/mike/Downloads/reinforelearn/snack/ckpt/'
        self.sess.run(self.initiation)
        if tf.train.get_checkpoint_state(self.ckptdir):
            self.saver.restore(self.sess, save_path=self.ckptdir)

    def train(self, trainlist):
        a = pd.DataFrame(trainlist)

        trainlist = a.values.tolist()
        self.timer += 1
        data = pd.DataFrame(trainlist, columns=['a', 'b', 'c', 'd','e'])
        data = data.drop_duplicates()
        data['f'] = data['g'] = data['h'] = np.nan
        data.loc[data.e == 'up', ['e','f','g','h']] = [1,0,0,0]
        data.loc[data.e == 'down', ['e','f','g','h']] = [0,1,0, 0]
        data.loc[data.e == 'left', ['e','f','g','h']] = [0,0,1, 0]
        data.loc[data.e == 'right', ['e','f','g','h']] = [0,0,0,1]

        x_data = np.array(data.iloc[:, :4], dtype=np.float32)
        y_data = np.array(data.iloc[:, 4:], dtype=np.float32)
        self.trainlist = x_data
        self.y_data = y_data


        self.sess.run(self.optimizer, feed_dict={self.x: x_data, self.y: y_data})
        if self.timer % 50 == 0:
            print('epoch: ', self.timer)
            self.saver.save(self.sess, save_path=self.ckptdir)
            # print(self.sess.run(tf.argmax(self.sess.run(self.logits, feed_dict={self.x: [[20,30,20,50]]}), 1)))
            # print(self.sess.run(tf.argmax(self.sess.run(self.logits, feed_dict={self.x: [[4, 3]]}), 1)))

    def evaluate(self, state):
        a = self.sess.run(self.logits, feed_dict={self.x: [state]})
        #print(self.sess.run(self.w1))
        # print(a)
        return self.sess.run(tf.argmax(a, 1))
