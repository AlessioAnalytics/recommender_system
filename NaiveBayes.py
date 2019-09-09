from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import numpy as np
import pandas as pd
import json
from tools.load import loader
import sys
from serve_data import data_server
import joblib


class NaiveBayes:

    def __init__(self, config={}):
        self.config = config

    def get_model(self):

        if self.config["NaiveBayes"]["model_type"] == "multinomial":
            model = MultinomialNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
        elif self.config["NaiveBayes"]["model_type"] == "bernoulli":
            model = BernoulliNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
        elif self.config["NaiveBayes"]["model_type"] == "complement":
            model = ComplementNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
        elif self.config["NaiveBayes"]["model_type"] == "gaussian":
            model = GaussianNB(alpha=self.config["NaiveBayes"]["alpha"], fit_prior=self.config["NaiveBayes"]["fit_prior"])
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

                    if self.config["n_train_batches"] == 1:
                        model.fit(x, t)
                    else:
                        model.partial_fit(x, t, classes)

            title = self.config["dataset"] + "_model"
            joblib.dump(model, self.config["dataset"]+"/models/"+title+".model")

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

                model.fit(x, t)

                title = self.config["dataset"] + "_model_no_" + str(index)
                joblib.dump(model, self.config["dataset"] + "/models/" + title+".model")


    def predict(self, ds=0):
        if self.config["approach"] == "multi":
            x = ds.get_value("KPM", self.config["pred_set"])

            if self.config["use_user_info"]:
                x = np.hstack((x, ds.get_value("info", self.config["pred_set"])))

            title = self.config["dataset"] + "_model"
            model = joblib.load(self.config["dataset"] + "/models/" + title + ".model")

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
                model = joblib.load(self.config["dataset"] + "/models/" + title +".model")
                prediction[:, index] = model.predict_proba(x)[:, 1]

        title = self.config["dataset"] + "_predictions_" + "fit" + self.config["fit_set"] + \
                "_pred" + self.config["pred_set"] + "_" + self.config["NaiveBayes"]["model_type"] + \
                "_approach" + str(self.config["approach"]) + "_split" + self.config["split"] + \
                "_count" + str(self.config["count"]) + \
                "_info" + str(self.config["use_user_info"]) + self.config["info_string"]

        np.save(self.config["dataset"]+"/npy_files/"+title, prediction)
        self.config["n_pred_batches"] = ds.save_batches(data=[prediction],
                                         names=["prediction"],
                                         batch_size=self.config["pred_batch_size"])

        with open(self.config["dataset"] + "/json_files/config.json", "w") as fp:
            json.dump(self.config, fp, indent=5)


def do(dataset):
    print("load config")
    with open(dataset + "/json_files/config.json", 'r') as fp:
        config = json.load(fp)
    print("start making the model")
    nb = NaiveBayes(config)
    ds = data_server(config,calc_vals=False)
    print("start training")
    nb.fit(ds=ds)
    print("training finishes")
    print("start prediction")
    nb.predict(ds=ds)
    print("prediction finished")


if __name__ == "__main__":
    dataset = sys.argv[1]
    do(dataset)
