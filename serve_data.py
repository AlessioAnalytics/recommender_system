import sys
import pandas as pd
import numpy as np
from tools.load import loader
import retrieve_information
import json
import os


class data_server:

    def __init__(self, config, calc_vals= True):
        self.config = config
        if self.config["use_user_info"]:
            if self.config["user_features"][0] != self.config["col_name_list"][0]:
                self.config["user_features"] = self.config[
                    "user_features"]  # [self.config["col_name_list"][0]]+ self.config["user_features"]
                self.config["user_features_prep_types"] = self.config[
                    "user_features_prep_types"]  # ["pass"]+self.config["user_features_prep_types"]
            for feature in self.config["user_features"][1:]:
                self.config["info_string"] += "_" + feature
        else:
            self.config["user_features"] = []
            self.config["user_features_prep_types"] = []
            self.config["n_info_cols"] = 0
            self.config["info_string"] = ""

        if calc_vals:
            self.save_needed_vals_to_rom()

    def save_needed_vals_to_rom(self):
        """
        check which values are needed for the given configuration
        then save them in rom to the get_value() funktion can serve them
        values: KPM, info and indexes
        :return: nothing
        """
        ir_exists = False
        if self.config["n_Kunden"] == 0 and self.config["n_Produkte"] == 0:
            print("data_keys", self.config["col_name_list"] + self.config["user_features"])
            ir = retrieve_information.information_retriever(
                dataset=self.config["dataset"],
                file_name=self.config["file_name_list"][0],
                data_keys=list(self.config["col_name_list"] + self.config["user_features"]),
                prep_types=["indexing", "indexing"] + self.config["user_features_prep_types"],
                save_lookups=True,
                save_dataset_info=True)

            self.config["n_Kunden"] = ir.dataset_info["keys"][self.config["col_name_list"][0]]["n_unique"]
            self.config["n_Produkte"] = ir.dataset_info["keys"][self.config["col_name_list"][1]]["n_unique"]
            ir_exists = True

        with open(self.config["dataset"]+"/json_files/dataset_info.json","r") as fp:
            full_dataset_info = json.load(fp)

        if self.config["split"] == "orders":
            for dset in [self.config["fit_set"], self.config["pred_set"]]:
                if not os.path.isfile(self.config["dataset"] + "/npy_files/" + dset + "_KPM.npy"):
                    set_ir = self.make_set_KPM(dset)

                if self.config["use_user_info"]:
                    if os.path.isfile(self.config["dataset"] + "/npy_files/" + dset + "_KPM.npy"):
                        set_ir = retrieve_information.information_retriever(
                            dataset=self.config["dataset"],
                            file_name=self.config["file_name_list"][np.argwhere(
                                np.array(["full", "train", "test", "valid"]) == dset)[0, 0]],
                            data_keys=list(self.config["col_name_list"] + self.config["user_features"]),
                            prep_types=["indexing", "indexing"] + self.config["user_features_prep_types"],
                            save_lookups=False,
                            save_dataset_info=False)


                    test= []
                    col_keys = []
                    missing_col_indexes = []
                    for key in self.config["user_features"]:
                        test = test + [dinf in set_ir.dataset_info["keys"][key]["col_keys"] for dinf in full_dataset_info["keys"][key]["col_keys"]]
                        col_keys = col_keys + full_dataset_info["keys"][key]["col_keys"]

                    for i in range(len(test)):
                        if not test[i]:
                            missing_col_indexes.append(i)
                            print("no sample of user feature class '" +
                                  col_keys[i] + "' in Dataset '" + dset + "'")

                    client_data = pd.DataFrame(set_ir.col_dict).values[:,
                                  [0] + [n + 2 for n in range(len(set_ir.col_dict.keys()) - 2)]]

                    self.match_clients(client_data, dset, missing_col_indexes)

        if self.config["split"] == "clients":
            if not os.path.isfile(self.config["dataset"] + "/npy_files/full_KPM.npy"):
                ir = self.make_set_KPM("full")
                ir_exists = True

            if self.config["use_user_info"]:
                if os.path.isfile(self.config["dataset"] + "/npy_files/full_KPM.npy") and not ir_exists:
                    ir = retrieve_information.information_retriever(
                        dataset=self.config["dataset"],
                        file_name=self.config["file_name_list"][0],
                        data_keys=list(self.config["col_name_list"] + self.config["user_features"]),
                        prep_types=["indexing", "indexing"] + self.config["user_features_prep_types"],
                        save_lookups=False,
                        save_dataset_info=False)

                client_data = pd.DataFrame(ir.col_dict).values[:,
                              [0] + [n + 2 for n in range(len(ir.col_dict.keys()) - 2)]]
                self.match_clients(client_data, "full")

            for dset in [self.config["fit_set"], self.config["pred_set"]]:
                if not os.path.isfile(self.config["dataset"] + "/npy_files/" + dset + "_index.npy"):
                    self.split_indices()

    def match_clients(self, data, title,missing_col_indexes = []):
        """
        matches all orders from the same clients and saves an Avarage client based on the give n data
        :param data: data which should be matched, the unique values in the first column will be matched
                        so the first column should be the client index !!
        :param title: title of the info file, in this context it will be the set [full train test valid ]
        :return: nothing just save the file to save ram
        """

        n_Aufträge, n_features = np.shape(data)
        info = np.zeros((self.config["n_Kunden"], n_features - 1))
        count = np.zeros(self.config["n_Kunden"])

        for dat in data:
            index = int(dat[0])
            count[index] += 1
            info[index] = info[index] + dat[1:]

        for i in range(self.config["n_Kunden"]):
            info[i] = info[i] / count[i] if count[i] != 0 else info[i]

        for index, number in zip(missing_col_indexes,np.arange(len(missing_col_indexes))):
            info = np.insert(info,index+number,0,axis = 1)

        np.save(self.config["dataset"] + "/npy_files/" + title + "_info.npy", info)

    def split_indices(self):
        """
        save sets [full train test valid ] of clients indices and shuffle to split the data along clients
        :return: nothing just save the files containingthe shuffeled indices
        """
        shuffled_indexes = np.arange(self.config["n_Kunden"])
        np.random.shuffle(shuffled_indexes)
        split_indices = np.round(np.array(self.config["split_ratio"]) * self.config["n_Kunden"] + 0.5).astype(int)
        split_indices[1] += split_indices[0]
        split_indices[2] += split_indices[1]
        train_data = shuffled_indexes[:split_indices[0]]
        test_data = shuffled_indexes[split_indices[0]:split_indices[1]]
        valid_data = shuffled_indexes[split_indices[1]:split_indices[2]]
        np.save(self.config["dataset"] + "/npy_files/full_index.npy", np.arange(self.config["n_Kunden"]))
        np.save(self.config["dataset"] + "/npy_files/train_index.npy", train_data)
        np.save(self.config["dataset"] + "/npy_files/test_index.npy", test_data)
        np.save(self.config["dataset"] + "/npy_files/valid_index.npy", valid_data)

    def make_set_KPM(self, dset):
        """
        make a KPM for the needed set [full train test valid ]
            usage of an information retriever object
        :param dset: set in [full train test valid ]
        :return: information retriever object
        """
        file_name = self.config["file_name_list"][
            np.argwhere(np.array(["full", "train", "test", "valid"]) == dset)[0, 0]]
        data_keys = list(self.config["col_name_list"])
        prep_types = ["indexing", "indexing"]
        if self.config["use_user_info"]:
            data_keys += self.config["user_features"]
            prep_types += self.config["user_features_prep_types"]

        set_ir = retrieve_information.information_retriever(dataset=self.config["dataset"],
                                                            file_name=file_name,
                                                            data_keys=data_keys,
                                                            prep_types=prep_types,
                                                            save_lookups=False,
                                                            save_dataset_info=False)
        count = self.config["count"]
        self.config["count"] = True
        np.save(self.config["dataset"] + "/npy_files/" + dset + "_KPM.npy",
                self.get_KPM(pd.DataFrame(set_ir.col_dict).values[:, :2]))
        self.config["count"] = count
        return set_ir

    def get_KPM(self, data):
        """
        transform indexed table with 2 columns to a 2d Matrix
        :param data: indexed table
        :return: 2d Matrix
        """
        Kunden_Produkte_Matrix = np.zeros((self.config["n_Kunden"], self.config["n_Produkte"]))
        for dat in data:
            if self.config["count"]:
                Kunden_Produkte_Matrix[int(dat[0]), int(dat[1])] += 1
            else:
                Kunden_Produkte_Matrix[int(dat[0]), int(dat[1])] = 1
        return Kunden_Produkte_Matrix

    def save_batches(self, data=[], names=[], batch_size=100):
        """
        save a list of data into batches
        :param data: list of data
        :param names: list of names how the batches of each data in the datalist will be called
               dir: self.config["dataset"] + "/batches/"+names[index]+"_batch_no_" + str(batch_index) + ".npy"
        :param batch_size: size of the batches
        :return: number of batches
        """
        N = len(data)

        len_check = len(data[0])
        for dat in data[1:]:
            if not len_check == len(dat):
                print(
                    "shapes dont fit for method save_batches. data should be a tuple of iterables of the same length ")
                sys.exit(0)

        n_batches = round(len_check / batch_size + 0.5)
        for batch_index in range(n_batches):

            if self.config["show_progress"]:
                if batch_index == 0:
                    load = loader(n_batches, "save_batches")
                load.print_progress(batch_index)
            if batch_index == n_batches - 1:
                batch_data = [dat[batch_index * batch_size:] for dat in data]
            else:
                batch_data = [dat[batch_index * batch_size:(batch_index + 1) * batch_size] for dat in data]
            for index in range(N):
                np.save(self.config["dataset"] + "/batches/" + names[index] + "_batch_no_" + str(
                    batch_index) + ".npy", batch_data[index])

        return n_batches

    def get_value(self, val_type, dset):
        """
        load the needed value from rom
        :param val_type: what value [KPM info index]
        :param dset: which set [full train test valid ]
        :return: value
        """
        if self.config["split"] == "orders":
            if val_type == "KPM":
                value = np.load(self.config["dataset"] + "/npy_files/" + dset + "_KPM.npy")
                if not self.config["count"]:
                    value = np.sign(value)
            if val_type == "info":
                value = np.load(self.config["dataset"] + "/npy_files/" + dset + "_info.npy")
            if val_type == "indexes":
                value = np.arange(self.config["n_Kunden"])

        elif self.config["split"] == "clients":
            indexes = np.load(self.config["dataset"] + "/npy_files/" + dset + "_index.npy")
            if val_type == "KPM":
                value = np.load(self.config["dataset"] + "/npy_files/full_KPM.npy")[indexes]
                if not self.config["count"]:
                    value = np.sign(value)
            if val_type == "info":
                value = np.load(self.config["dataset"] + "/npy_files/full_info.npy")[indexes]
            if val_type == "indexes":
                value = indexes
        return value

    def get_training_data(self, train_set):
        """
        generatie training data based on given configuration
        :param train_set:
        :return:
        """
        if self.config["use_user_info"]:
            user_info = self.get_value("info", train_set)
            self.config["n_info_cols"] = len(user_info[0])

        KPM = self.get_value("KPM", train_set)
        n_k, n_p = KPM.shape

        if self.config["approach"] == "multi":
            target = [i for i in range(self.config["n_Produkte"])]
            data = [[0 for i in range(self.config["n_Produkte"] + self.config["n_info_cols"])] for i in
                    range(self.config["n_Produkte"])]

            for kunden_index in range(n_k):
                if self.config["show_progress"]:
                    if kunden_index == 0:
                        load = loader(n_k, "save_batches")
                    load.print_progress(kunden_index)

                for produkt_index in np.argwhere(KPM[kunden_index] > 0)[:, 0]:
                    target.append(produkt_index)
                    var_Kunde = np.array(KPM[kunden_index])
                    var_Kunde[produkt_index] = 0
                    if self.config["use_user_info"]:
                        var_Kunde = np.hstack((var_Kunde, user_info[kunden_index]))
                    data.append(var_Kunde)

            self.config["n_train_batches"] = self.save_batches(data=(data, target),
                                                                names=["data", "target"],
                                                                batch_size=self.config["train_batch_size"])

        elif self.config["approach"] == "binary":
            for prod_n in range(self.config["n_Produkte"]):
                if self.config["show_progress"]:
                    if prod_n == 0:
                        load = loader(full=self.config["n_Produkte"], message="save_batches")
                    load.print_progress(prod_n)

                target = KPM[:, prod_n]
                data = np.delete(KPM, prod_n, axis=1)
                if self.config["use_user_info"]:
                    data = np.hstack((data, user_info))

                show_progress = self.config["show_progress"]
                self.config["show_progress"] = False
                self.config["n_train_batches"] = self.save_batches(data=[data, target],
                                                                   names=["data_model_" + str(prod_n),
                                                                          "target_model_" + str(prod_n)],
                                                                   batch_size=self.config["train_batch_size"])

                self.config["show_progress"] = show_progress

def do(dataset):
    print("load config")
    if os.path.exists(dataset + "/json_files/config.json") is not None:
        with open(dataset + "/json_files/config.json", "r") as fp:
            config = json.load(fp)
    print("make object")
    ds = data_server(config)

    if config["approach"] == "multi":
        print("making training data")
        ds.get_training_data(config["fit_set"])
        print("finished making training data")
    else:
        ds.config["n_train_batches"] = 1

    print("start saving prediction info batches")
    ds.config["n_pred_batches"] = ds.save_batches(
        data=[ds.get_value("KPM", config["pred_set"]), ds.get_value("indexes", config["pred_set"])],
        names=["KPM", "indexes"],
        batch_size=config["pred_batch_size"])

    print("finished saving prediction info batches")
    with open(ds.config["dataset"] + "/json_files/config.json", "w") as fp:
        json.dump(ds.config, fp, indent=5)


if __name__ == "__main__":
    dataset = sys.argv[1]
    do(dataset)
