from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import os
from luwi_net_class import LuWi_Network
import json
from tools.load import loader
import sys
from serve_data import data_server
import joblib


class MachineLearning:

    def __init__(self, config={}):
        self.config = config
        self.ds = ds

    def get_model(self):
        if self.config["model"]=="DeepLearning":
            model = self.make_deep_learning_model()
        elif self.config["model"]=="NaiveBayes":
            model = self.make_naive_bayes_model()
        elif self.config["model"]=="LuwiNet":
            model = self.make_luwi_net_model()
        return model

    def make_naive_bayes_model(self):
        if self.config["NaiveBayes"]["model_type"] == "multinomial":
            model = MultinomialNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
        elif self.config["NaiveBayes"]["model_type"] == "bernoulli":
            model = BernoulliNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
        elif self.config["NaiveBayes"]["model_type"] == "complement":
            model = ComplementNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
        elif self.config["NaiveBayes"]["model_type"] == "gaussian":
            model = GaussianNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
        return model

    def make_luwi_net_model(self,n_train_Kunden,n_Produkte):

        model = LuWi_Network(model_name=self.config["model_name"],
                             input1_dim=len(self.ds.get_value("indexes",self.config["fit_set"])),
                             input2_dim=self.config["n_Produkte"],
                             embedding_size=self.config["embedding_size"],
                             learning_rate=self.config["learning_rate"],
                             hidden_dims1=self.config["hidden_dims1"],
                             hidden_dims2=self.config["hidden_dims2"]),
        model.make_graph()
        return model

    def make_deep_learning_model(self):
        if self.config["approach"] == "binary":
            threshold = 0.5
        elif self.config["approach"] == "multi":
            threshold = 1/self.config["n_Produkte"]
        if os.path.isfile(self.config["dataset"] + "/models/DeepLearning/models/"+self.config["model_name"] + ".h5"):

            model = models.load_model(self.config["dataset"] + "/models/DeepLearning/models/"+self.config["model_name"] + '.h5')

            # Load weights into the new model
            model.load_weights(self.config["dataset"] + "/models/DeepLearning/weights/"+self.config["DeepLearning"]["weights_name"] + '.h5')

            print("model " + self.config["model_name"] + " loaded")
        else:
            out_dim = self.config["n_Produkte"]
            in_dim = int(self.config["n_Produkte"]+self.config["n_info_cols"])
            if self.config["approach"]=="binary":
                in_dim -= 1
                out_dim = 2

            model = models.Sequential([
                layers.Dense(int(1.2*self.config["n_Produkte"]), input_shape=(in_dim,)),
                layers.Activation('relu'),
                layers.Dense(int(1.5*self.config["n_Produkte"])),
                layers.Activation('relu'),
                layers.Dense(int(1.2*self.config["n_Produkte"])),
                layers.Activation('relu'),
                layers.Dense(int(1 * out_dim)),
                layers.Activation('softmax'),
            ])
            print("input_dimension",in_dim)
            print("output_dimension", out_dim)
            model.summary()
            print("model " + self.config["model_name"] + " made")
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.Precision(thresholds=threshold),tf.keras.metrics.Recall(thresholds=threshold)])

        return model

    def save_model(self,title,model):
        if self.config["model"] == "NaiveBayes":
            joblib.dump(model, self.config["dataset"] + "/models/" + title + ".model")
        elif self.config["model"] == "DeepLearning":
            model.save_weights(self.config["dataset"] + "/models/DeepLearning/weights/"+ self.config["DeepLearning"]["weights_name"] + '.h5')
            model.save(self.config["dataset"] + "/models/DeepLearning/models/"+self.config["model_name"] + ".h5")

    def load_model(self,title):
        if self.config["approach"] == "binary":
            threshold = 0.5
        elif self.config["approach"] == "multi":
            threshold = 1 / self.config["n_Produkte"]
        if self.config["model"] == "NaiveBayes":
            model = joblib.load(self.config["dataset"] + "/models/" + title + ".model")
        elif self.config["model"] == "DeepLearning":
            model = models.load_model(self.config["dataset"] + "/models/DeepLearning/models/"+self.config["model_name"] + '.h5')

            # Load weights into the new model
            model.load_weights(self.config["dataset"] + "/models/DeepLearning/weights/"+self.config["DeepLearning"]["weights_name"] + '.h5')
            model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=[tf.keras.metrics.Precision(thresholds=threshold),tf.keras.metrics.Recall(thresholds=threshold)])
            print("model " + self.config["model_name"] + " loaded")
        return model

    def fit(self, names=["data", "target"], ds=0):

        if self.config["approach"] == "multi":
            model = self.get_model()
            classes = np.arange(self.config["n_Produkte"])
            if self.config["n_train_batches"]:
                for batch_index in range(self.config["n_train_batches"]):
                    if self.config["show_progress"]:
                        if batch_index == 0:
                            load = loader(self.config["n_train_batches"], "train_batches")
                        load.print_progress(batch_index)

                    x = np.load(self.config["dataset"] + "/batches/" +
                                names[0] + "_batch_no_" + str(batch_index) + ".npy")
                    t = np.load(self.config["dataset"] + "/batches/" +
                                names[1] + "_batch_no_" + str(batch_index) + ".npy")

                    if self.config["model"]=="DeepLearning":
                        T = np.zeros((len(t), self.config["n_Produkte"]))
                        for row,col in zip(np.arange(len(t)),t):
                            T[row,col] = 1
                        model.fit(x,T,epochs=self.config["DeepLearning"]["n_epochs"] ,verbose = True)
                    else:
                        if self.config["n_train_batches"] == 1:
                            model.fit(x, t)
                        else:
                            model.partial_fit(x, t, classes)

            title = self.config["dataset"] + "_model"
            self.save_model(title,model)

        elif self.config["approach"] == "binary":
            for index in range(self.config["n_Produkte"]):
                model = self.get_model()
                classes = np.array([0, 1])

                if self.config["show_progress"]:
                    if index == 0:
                        load = loader(self.config["n_Produkte"], "train")
                    load.print_progress(index)

                KPM = ds.get_value("KPM", self.config["fit_set"])

                if self.config["use_user_info"]:
                    KPM = np.hstack((KPM, ds.get_value("info", self.config["fit_set"])))

                x = np.delete(KPM, index,axis=1)
                t = KPM[:, index]

                if self.config["model"] == "DeepLearning":
                    T = np.zeros((len(t),2))
                    for row, col in zip(np.arange(len(t)), t):
                        T[row, int(col)] = 1
                    model.fit(x, T, epochs=self.config["DeepLearning"]["n_epochs"], verbose=True)
                elif self.config["model"] == "NaiveBayes":
                    model.fit(x, t)


                title = self.config["dataset"] + "_model_no_" + str(index)
                self.save_model(title,model)



    def predict(self, ds=0):
        if self.config["approach"] == "multi":
            x = ds.get_value("KPM", self.config["pred_set"])

            if self.config["use_user_info"]:
                x = np.hstack((x, ds.get_value("info", self.config["pred_set"])))

            title = self.config["dataset"] + "_model"
            model = self.load_model(title)

            if self.config["model"] == "DeepLearning":
                prediction = model.predict(x)
            elif self.config["model"] == "NaiveBayes":
                prediction = model.predict_proba(x)

        elif self.config["approach"] == "binary":
            KPM = ds.get_value("KPM", self.config["pred_set"])
            prediction = np.zeros_like(KPM)

            if self.config["use_user_info"]:
                KPM = np.hstack((KPM, ds.get_value("info",self.config["pred_set"])))

            for index in range(self.config["n_Produkte"]):
                if self.config["show_progress"]:
                    if index == 0:
                        load = loader(self.config["n_Produkte"], "predict")
                    load.print_progress(index)

                x = np.delete(KPM, index, axis=1)
                title = self.config["dataset"] + "_model_no_" + str(index)
                model = self.load_model(title)

                if self.config["model"] == "DeepLearning":
                    prediction[:, index] = model.predict(x)[:, 1]
                elif self.config["model"] == "NaiveBayes":
                    prediction[:, index] = model.predict_proba(x)[:, 1]

        # title = self.config["dataset"] + "_predictions_" + "fit" + self.config["fit_set"] + \
        #         "_pred" + self.config["pred_set"] + "_" + self.config["NaiveBayes"]["model_type"] + \
        #         "_approach" + str(self.config["approach"]) + "_split" + self.config["split"] + \
        #         "_count" + str(self.config["count"]) + \
        #         "_info" + str(self.config["use_user_info"]) + self.config["info_string"]

        title = self.config["model_name"]+"_predictions"
        np.save(self.config["dataset"]+"/npy_files/"+title, prediction)
        self.config["n_pred_batches"] = ds.save_batches(data=[prediction],
                                         names=["prediction"],
                                         batch_size=self.config["pred_batch_size"])

        with open(self.config["dataset"] + "/json_files/config.json", "w") as fp:
            json.dump(self.config, fp, indent=5)


def do(dataset):
    print("load config")
    with open( "{dataset}/json_files/config.json".format(dataset=dataset), 'r') as fp:
        config = json.load(fp)
    print(config["dataset"])
    print("start making the model")
    ds = data_server(config, calc_vals=False)
    ml = MachineLearning(config,ds)

    print("start training")
    ml.fit(ds=ds)
    print("training finishes")
    print("start prediction")
    ml.predict(ds=ds)
    print("prediction finished")


if __name__ == "__main__":
    dataset = sys.argv[1].strip()
    do(dataset)
