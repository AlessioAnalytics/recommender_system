from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import sys
import pandas as pd
import numpy as np
from tools import load
from tools import npy_to_csv
from retrieve_information import get_info, get_orders, get_lookups
import json
from sklearn.utils import shuffle
import os


class NaiveBayes:

    def __init__(self,
                 dataset="Superstore",
                 file_name_list=["Superstore",
                                 "Superstore_train",
                                 "Superstore_test",
                                 "Superstore_valid"],

                 col_name_list=["Kundenname", "Produktname"],
                 model_type="multinomial",
                 show_progress=False,
                 use_user_info=True,
                 user_features=["Kundenname", "Kategorie", "Segment", "Land"],
                 user_features_prep_types=["pass", "one_hot", "one_hot", "one_hot"],
                 fit_set="train",
                 approach=2,
                 split="orders",
                 make_model=True,
                 batch_learning=True,
                 batch_size=5000):

        self.file_name_list = file_name_list
        self.dataset = dataset
        self.col_name_list = col_name_list

        self.model_type = model_type
        self.approach = approach
        self.show_progress = show_progress

        self.batch_learning = batch_learning
        self.batch_size = batch_size

        if split in ["clients", "orders"]:
            self.split = split
        else:
            raise ValueError('split should be either "orders" or "clients"')

        if fit_set in ["full", "train", "test", "valid"]:
            self.fit_set = fit_set
        else:
            raise ValueError('fit_set should be "full","train","test" or "valid"')

        self.use_user_info = use_user_info
        self.user_features = user_features
        self.user_features_prep_types = user_features_prep_types

        self.info_string = ""
        for feature in self.user_features[1:]:
            self.info_string += "_"
            self.info_string += feature

        if make_model:
            get_lookups(dataset, file_name_list[0], col_name_list, ["indexing", "indexing"])

            self.n_Kunden = int(np.max(
                pd.read_csv(self.dataset + "/csv_files/" + col_name_list[0] + "_lookup.csv", sep="\t").values[:,
                0])) + 1
            self.n_Produkte = int(np.max(
                pd.read_csv(self.dataset + "/csv_files/" + col_name_list[1] + "_lookup.csv", sep="\t").values[:,
                0])) + 1

            self.full_KPM = self.get_KPM(
                get_orders(self.dataset, self.file_name_list[0], self.col_name_list, ["indexing", "indexing"]))

            if self.use_user_info:
                self.make_info_dict()

            if self.split == "clients":
                self.make_client_split_dict()
                for type in ["full", "train", "test", "valid"]:
                    for field in ["KPM", "info", "indexes"]:
                        np.save("Alteryx/npy_files/client_split_dict_" + type + "_" + field + ".npy",
                                self.client_split_dict[type][field])
            else:
                self.make_KPM_dict(save=True)

            if self.approach == 1:
                self.make_model_approach_1(fit_set)
            elif self.approach == 2:
                self.make_model_approach_2(fit_set)

    def get_data(self, file_name, col_names=["Kundenname_index", "Produktname_index"]):
        return pd.read_csv(self.dataset + "/csv_files/" + file_name + ".csv", sep="\t")[col_names].values

    def get_KPM(self, data):
        Kunden_Produkte_Matrix = np.zeros((self.n_Kunden, self.n_Produkte))
        for dat in data:
            Kunden_Produkte_Matrix[int(dat[0]), int(dat[1])] = 1
        return Kunden_Produkte_Matrix

    def make_KPM_dict(self, save=False):
        self.KPM_dict = {}
        for name, type in zip(self.file_name_list, ["full", "train", "test", "valid"]):
            data = get_orders(self.dataset, name, self.col_name_list, ["indexing", "indexing"])
            KPM = self.get_KPM(data)
            self.KPM_dict[type] = KPM
            if save:
                np.save(self.dataset + "/npy_files/KPM_" + type, KPM)

    def make_info_dict(self):
        self.info_dict = {}
        for name, type in zip(self.file_name_list, ["full", "train", "test", "valid"]):
            info = get_info(self.dataset, name, self.user_features, self.user_features_prep_types)
            self.info_dict[type] = info

    def make_client_split_dict(self, split=(0.7, 0.2, 0.1), save=False):
        KPM = self.full_KPM
        if self.use_user_info:
            info = self.info_dict["full"]
        else:
            info = np.zeros((self.n_Kunden, 1))

        if os.path.isfile(self.dataset + "/npy_files/" + "client_split_full_indexes" + ".npy"):
            self.client_split_dict = {}
            for type in ["full", "train", "test", "valid"]:
                self.client_split_dict[type] = {"KPM": KPM[list(
                    np.load(self.dataset + "/npy_files/" + "client_split_" + type + "_indexes" + ".npy")), :],
                                                "info": info[list(np.load(
                                                    self.dataset + "/npy_files/" + "client_split_" + type + "_indexes" + ".npy")),
                                                        :],
                                                "indexes": np.load(
                                                    self.dataset + "/npy_files/" + "client_split_" + type + "_indexes" + ".npy")}
        else:
            indexes, KPM, info = shuffle(np.arange(self.n_Kunden), KPM, info)

            split_indices = np.round(np.array(split) * self.n_Kunden + 0.5).astype(int)
            split_indices[1] += split_indices[0]
            split_indices[2] += split_indices[1]

            self.client_split_dict = {"full": {"KPM": KPM,
                                               "info": info,
                                               "indexes": indexes},
                                      "train": {"KPM": KPM[:split_indices[0]],
                                                "info": info[:split_indices[0]],
                                                "indexes": indexes[:split_indices[0]]},
                                      "test": {"KPM": KPM[split_indices[0]:split_indices[1]],
                                               "info": info[split_indices[0]:split_indices[1]],
                                               "indexes": indexes[split_indices[0]:split_indices[1]]},
                                      "valid": {"KPM": KPM[split_indices[1]:],
                                                "info": info[split_indices[1]:],
                                                "indexes": indexes[split_indices[1]:]}
                                      }
            for type in ["full", "train", "test", "valid"]:
                np.save(self.dataset + "/npy_files/" + "client_split_" + type + "_indexes" + ".npy",
                        sorted(self.client_split_dict[type]["indexes"]))

    def make_model_approach_1(self, set="full"):

        if self.use_user_info:
            if self.split == "clients":
                user_info = self.client_split_dict[set]["info"]
            else:
                user_info = self.info_dict[set]
            n_user_features = len(user_info[0])
        else:
            n_user_features = 0

        if self.split == "clients":
            KPM = self.client_split_dict[set]["KPM"]
        else:
            KPM = self.KPM_dict[set]

        self.model_list = []
        for prod_n in range(self.n_Produkte):
            if self.show_progress:
                load.print_progress(prod_n / self.n_Produkte, "make_model")


            target = KPM[:, prod_n]
            data = np.delete(KPM, prod_n, axis=1)

            if self.use_user_info:
                data = np.hstack((data, user_info))

            if self.model_type == "multinomial":
                model = MultinomialNB()
            elif self.model_type == "bernoulli":
                model = BernoulliNB()
            elif self.model_type == "complement":
                model = ComplementNB()
            elif self.model_type == "gaussian":
                model = GaussianNB()

            model.fit(data, target)
            self.model_list.append(model)

    def make_model_approach_2(self, set="full"):

        if self.use_user_info:
            if self.split == "clients":
                user_info = self.client_split_dict[set]["info"]
            else:
                user_info = self.info_dict[set]
            n_user_features = len(user_info[0])
        else:
            n_user_features = 0

        if self.split == "clients":
            KPM = self.client_split_dict[set]["KPM"]
        else:
            KPM = self.KPM_dict[set]
        n_k, n_p = KPM.shape

        target = []
        data = []
        # target = [i for i in range(self.n_Produkte)]
        # data = [[0 for i in range(self.n_Produkte+n_user_features)] for i in range(self.n_Produkte)]

        for kunden_index in range(n_k):
            if self.show_progress:
                load.print_progress(kunden_index / n_k, "prepare data")

            for produkt_index in np.argwhere(KPM[kunden_index] == 1)[:,
                                 0]:  # self.KPM_dict[set][kunden_index] == 1)[:,0]:
                target.append(produkt_index)
                var_Kunde = np.array(KPM[kunden_index])  # self.KPM_dict[set][kunden_index])
                var_Kunde[produkt_index] = 0
                if self.use_user_info:
                    var_Kunde = np.hstack((var_Kunde, user_info[kunden_index]))

                data.append(var_Kunde)

        print("target", len(target))  # target.shape)
        print("data", len(data))  # data.shape)

        if self.model_type == "multinomial":
            model = MultinomialNB()
        elif self.model_type == "bernoulli":
            model = BernoulliNB()
        elif self.model_type == "complement":
            model = ComplementNB()
        elif self.model_type == "gaussian":
            model = GaussianNB()

        print("a")
        if self.batch_learning:
            n_batches = self.save_batches(data, target)
            print("############################", n_batches, " ########################################")

            #			del data
            #			del target
            #			del var_Kunde
            #			del KPM
            #			gc.collect()
            print("now_train")
            model = self.train_batches(model, n_batches)
        else:
            model.fit(data, target)
        print("b")
        self.model = model

    def save_batches(self, data, target):
        n_batches = np.round(len(data) / self.batch_size + 0.5).astype(int)
        # save batches
        for batch_index in range(n_batches):
            if self.show_progress:
                load.print_progress(batch_index / n_batches, "save_batches")
            if batch_index == n_batches - 1:
                batch_data = data[batch_index * self.batch_size:]
                batch_target = target[batch_index * self.batch_size:]
            else:
                batch_data = data[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
                batch_target = target[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            np.save(self.dataset + "/batches/data_batch_no_" + str(batch_index) + ".npy", batch_data)
            np.save(self.dataset + "/batches/target_batch_no_" + str(batch_index) + ".npy", batch_target)
        return n_batches

    def train_batches(self, model, n_batches):
        for batch_index in range(n_batches):
            if self.show_progress:
                load.print_progress(batch_index / n_batches, "train_batches")
            batch_data = np.load(self.dataset + "/batches/data_batch_no_" + str(batch_index) + ".npy")
            batch_target = np.load(self.dataset + "/batches/target_batch_no_" + str(batch_index) + ".npy")
            model.partial_fit(batch_data, batch_target, classes=np.arange(self.n_Produkte))
        return model

    def predict_set_approach_2(self, set="test"):

        if self.split == "clients":
            kunden_vektor = self.client_split_dict[set]["KPM"]
        else:
            kunden_vektor = self.KPM_dict[set]

        if self.use_user_info:
            if self.split == "clients":
                kunden_vektor = np.hstack((kunden_vektor, self.client_split_dict[set]["info"]))
            else:
                kunden_vektor = np.hstack((kunden_vektor, self.info_dict[set]))

        prediction = self.model.predict_proba(kunden_vektor)
        return prediction

    def predict_set_approach_1(self, set="test"):

        if self.split == "clients":
            kunden_vektor = self.client_split_dict[set]["KPM"]
        else:
            kunden_vektor = self.KPM_dict[set]

        if self.use_user_info:
            if self.split == "clients":
                kunden_vektor = np.hstack((kunden_vektor, self.client_split_dict[set]["info"]))
            else:
                kunden_vektor = np.hstack((kunden_vektor, self.info_dict[set]))

        prediction = np.zeros_like(kunden_vektor)
        for index in range(self.n_Produkte):
            if self.show_progress:
                load.print_progress(index / self.n_Produkte, "Prediction")

            prediction[:, index] = self.model_list[index].predict_proba(np.delete(kunden_vektor, index, axis=1))[:, 1]
        return prediction

    def get_recommendations(self, n_recs=20):
        recs = np.zeros((self.n_Kunden, n_recs))
        for kunden_index in range(self.n_Kunden):
            recs[kunden_index] = np.array(
                sorted(zip(self.sk_client_prop[kunden_index], np.arange(self.n_Produkte)), reverse=True)[:n_recs])[:, 1]
        self.sk_recs = recs

    def export_as_csv_in_tableau_format(self, pred_set, predictions):
        n_Kunden, n_Produkte = predictions.shape
        dict = {"client": [], "content": [], "propability": [], "already_bought": []}
        for k in range(n_Kunden):
            if self.show_progress:
                load.print_progress(k / n_Kunden, "export")
            for p in range(n_Produkte):
                if self.split == "clients":
                    dict["client"].append(self.client_split_dict[pred_set]["indexes"][k])
                    dict["already_bought"].append(self.client_split_dict[pred_set]["KPM"][k, p])
                else:
                    dict["client"].append(k)
                    dict["already_bought"].append(self.KPM_dict[pred_set][k, p])
                dict["content"].append(p)
                dict["propability"].append(predictions[k, p])

        title = self.dataset + "_predictions_" + "fit" + self.fit_set + "_pred" + pred_set + "_" + self.model_type + "_approach" + str(
            self.approach) + "_split" + self.split + "_info" + str(self.use_user_info) + self.info_string
        pd.DataFrame(dict).to_csv("Tableau_exports/" + title + ".csv", index_label="Row_index", sep=";")


if __name__ == "__main__":
    if False: #sys.argv[1] == "usage":
        print("""
				Arguments:
				1. dataset -> Source folder
				2. file_name -> Source file name without ending(.csv)
				3. Client_col_name -> Name of the Client Column (input for NaiveBayes)
				4. Content_col_name -> Name of the Content Column (input for NaiveBayes)
				5. fit_set -> which set to train on
				6. pred_set -> which set to predict 
				7. split_type -> 	
					"orders": -> split train/test/valid data along the orders  
					"clients": -> split train/test/valid data along the clients 
				8. approach -> 2 possible approaches: 
					1: -> predict P(prod_x (Yes/No)|given prod_a...prod_n)					Binary Classification
					2: -> predict P(prod_x (prod_a or prod_b or ..)|given prod_a...prod_n)			faster Multiclass Classification 
				9. model_type choices ["multinomial","gaussian","beroulli","complement"] 
				
				10. use_user_info: if you want to use user features  set to 1 else 0
					after that type the feature/column-names after -user_features and the preprocessing types after -prep_types (has to be the same length!!!)
				
				example:  python3 NaiveBayes_sklearn.py Superstore Superstore Kundenname Produktname train test 2 multinomial 0 1 -user_features Segment Land -prep_types one_hot one_hot
								
				""")
        sys.exit(0)

    dataset = "Superstore"  # sys.argv[1]
    file_name = "Superstore"  # sys.argv[2]

    file_name_list = []
    for type in ["", "_train", "_test", "_valid"]:
        file_name_list.append(file_name + type)

    Client_col_name = "Kundenname" #"Cust. No."  # sys.argv[3]
    Product_col_name = "Produktname" #"Item No."  # sys.argv[4]
    fit_set = "full"  # sys.argv[5]
    pred_set = "full"  # sys.argv[6]
    split = "orders"  # sys.argv[7]
    approach = 1 # int(sys.argv[8])
    model_type = "multinomial"  # sys.argv[9]
    user_info = "0"  # sys.argv[10]
    args = []  # sys.argv[11:]

    user_features, user_features_prep_types = [], []
    l = []
    for arg in args:
        if arg == "-user_features":
            l = user_features
        elif arg == "-prep_types":
            l = user_features_prep_types
        else:
            l.append(arg)

    nb = NaiveBayes(dataset=dataset,
                    file_name_list=file_name_list,
                    col_name_list = [Client_col_name ,Product_col_name],  # ["CustomerID","ProductID"],
                    user_features=[Client_col_name ] +user_features,
                    user_features_prep_types=["pass" ] +user_features_prep_types,
                    show_progress=True,
                    split=split,
                    model_type=model_type,
                    use_user_info=bool(int(user_info)),
                    approach=approach,
                    fit_set=fit_set)

    if approach==1:
        prediction = nb.predict_set_approach_1(pred_set)

    if approach==2:
        prediction = nb.predict_set_approach_2(pred_set)
    np.save(dataset +"/npy_files/predictions" +"fit " +fit_set +"_pred " +pred_set +"" +nb.model_type +"_approach "
         +str(approach ) +"_split " +split +"_info " +str(nb.use_user_info ) +nb.info_string +".npy" ,prediction)
    nb.export_as_csv_in_tableau_format(pred_set ,prediction)

