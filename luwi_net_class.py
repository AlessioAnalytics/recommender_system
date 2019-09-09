import tensorflow as tf

import matplotlib.pyplot as plt
import os

class LuWi_Network:

    def __init__(self,model_name,input1_dim,input2_dim,embedding_size,learning_rate,hidden_dims1 = [], hidden_dims2 = []):
        self.model_name = model_name
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.hidden_dims1 = hidden_dims1
        self.hidden_dims2 = hidden_dims2

    def make_graph(self):
        self.g = tf.Graph()
        prod_dims = [self.input1_dim] + self.hidden_dims1 + [self.embedding_size]
        cust_dims = [self.input2_dim] + self.hidden_dims2 + [self.embedding_size]
        print("DIMS:", prod_dims, cust_dims)

        with self.g.as_default():
            self.tf_prod_in = tf.placeholder(shape=(None, self.input1_dim), dtype=tf.float32, name="tf_prod_in")
            self.tf_cust_in = tf.placeholder(shape=(None, self.input2_dim), dtype=tf.float32, name="tf_cust_in")
            self.tf_target = tf.placeholder(shape=(None, None), dtype=tf.float32, name="tf_target")

            tf_prod_weights_list = [tf.Variable(tf.random_normal([prod_dims[i], prod_dims[i + 1]])) for i in
                                    range(len(prod_dims) - 1)]
            tf_prod_bias_list = [tf.Variable(tf.random_normal([1, prod_dims[i + 1]])) for i in
                                 range(len(prod_dims) - 1)]

            tf_cust_weights_list = [tf.Variable(tf.random_normal([cust_dims[i], cust_dims[i + 1]])) for i in
                                    range(len(cust_dims) - 1)]
            tf_cust_bias_list = [tf.Variable(tf.random_normal([1, cust_dims[i + 1]])) for i in
                                 range(len(cust_dims) - 1)]
            var = self.tf_prod_in
            for i in range(len(prod_dims) - 1):
                var = tf.matmul(var, tf_prod_weights_list[i]) + tf_prod_bias_list[i]
                if i != len(prod_dims) - 2:
                    var = tf.nn.sigmoid(var)
                else:
                    var = tf.nn.softmax(var)
            tf_product_embedding = var

            var = self.tf_cust_in
            for i in range(len(cust_dims) - 1):
                var = tf.matmul(var, tf_cust_weights_list[i]) + tf_cust_bias_list[i]
                if i != len(cust_dims) - 2:
                    var = tf.nn.sigmoid(var)
                else:
                    var = tf.nn.softmax(var)
            tf_customer_embedding = var

            cust_abs = tf.sqrt(tf.reduce_sum(tf.math.square(tf.math.abs(tf_customer_embedding)), axis=1))
            prod_abs = tf.sqrt(tf.reduce_sum(tf.math.square(tf.math.abs(tf_product_embedding)), axis=1))
            abses = tf.tensordot(cust_abs, tf.transpose(prod_abs), axes=0)
            unnormalized_result = tf.matmul(tf_customer_embedding, tf.transpose(tf_product_embedding))
            self.result = tf.math.divide(unnormalized_result, abses, name="result")

            self.loss = tf.reduce_mean(tf.square(self.tf_target - self.result), name="loss")
            self.optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optim.minimize(self.loss, name="train_op")
            self.saver = tf.train.Saver()

    def fit(self, training_client_vectors, content_vectors,target ,n_epochs = 1000,learning_rate="None", test_client_vectors = "None", test_target = "None"):
        if learning_rate != "None":
            try:
                self.learning_rate = float(learning_rate)
                self.make_graph()
            except:
                print("invalid learning_rate")

        training_costs = []
        test_costs = []
        with tf.Session(graph=self.g) as sess:
            if os.path.isfile('./models/' + self.model_name+".meta"):
                self.saver.restore(sess, './models/' + self.model_name)
                print("model "+self.model_name+" restored")
            else:
                sess.run(tf.global_variables_initializer())
                print("Model "+self.model_name+" made")

            for e in range(n_epochs):
                train_c, _ = sess.run([self.loss, self.train_op],
                                feed_dict={self.tf_prod_in: content_vectors,
                                           self.tf_cust_in: training_client_vectors,
                                           self.tf_target: target
                                           }
                                )
                training_costs.append(train_c)


                if test_client_vectors != "None" and test_target != "None":
                    test_c = sess.run([self.loss],
                                    feed_dict={self.tf_prod_in: content_vectors,
                                               self.tf_cust_in: test_client_vectors,
                                               self.tf_target: test_target
                                               }
                                    )[0]
                    test_costs.append(test_c)

                if not e % int(n_epochs/100):
                    if test_client_vectors != "None" and test_target != "None":
                        print("Epoche %4d: train_error: %.30f test_error: %.30f" % (e, train_c, test_c))
                    else:
                        print("Epoche %4d: train_error: %.30f" % (e, train_c))


            saved_path = self.saver.save(sess, './models/'+self.model_name)
            plt.plot(training_costs)
            plt.title('training')
            plt.show()

            if test_client_vectors != "None" and test_target != "None":
                plt.plot(test_costs)
                plt.title('Test')
                plt.show()

    def predict(self,client_vectors, content_vectors):
        with tf.Session(graph=self.g) as sess:
            if os.path.isfile('./models/' +self.model_name + ".meta"):
                self.saver.restore(sess, './models/' + self.model_name)
                print("model "+self.model_name+" restored")
            else:
                sess.run(tf.global_variables_initializer())
                print("model "+self.model_name+" made")

            result = sess.run([self.result],
                           feed_dict={
                               self.tf_prod_in: content_vectors,
                               self.tf_cust_in: client_vectors,
                           })
        return result
