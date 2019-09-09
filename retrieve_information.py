from tools import Data_Preprocessor as prep
from tools.Data_Preprocessor import delete_duplicates
import numpy as np
import pandas as pd
import sys
import json


class information_retriever:

    def __init__(self, dataset="Superstore", file_name="Superstore",
                 data_keys=['Kundenname', 'Segment', 'Land', 'Region', 'Kategorie', 'Unterkategorie', 'Menge',
                            'Rabatt'],
                 prep_types=["pass", "one_hot", "one_hot", "one_hot", "one_hot", "one_hot", "normalize", "normalize"],
                 save_lookups=False, print_keys=False, save_dataset_info=True):

        self.file_name = file_name
        self.dataset = dataset
        self.data_keys = data_keys
        self.prep_types = prep_types
        self.save_dataset_info = save_dataset_info
        self.dataset_info   =   {
                                    "dataset": dataset,
                                    "file_name": file_name,
                                    "n_rows": file_name,
                                    "data_keys": data_keys,
                                    "prep_types": prep_types,
                                    "keys": {key: {"n_unique":0,"col_keys": []} for key in data_keys}
                                }
        self.key_dict = self.make_key_dict()
        self.col_dict = self.make_col_dict()

        if print_keys:
            self.print_keys()
        if save_lookups:
            self.save_lookups()

    def print_keys(self):
        """
        print all information extracted from the dataset
        :return:
        """
        for key in self.data_keys:
            print("Key", key)
            print("Prep_Type", self.key_dict[key]["prep_type"])
            print("Uniques", len(self.key_dict[key]["lookup"]))
            print("Columns", self.key_dict[key]["col_keys"])

    def save_lookups(self):
        """
        save the lookup tables as csv file
        :return:
        """
        for key_index in range(len(self.prep_types)):
            if self.prep_types[key_index] == "indexing":
                df = pd.DataFrame({self.data_keys[key_index]: self.key_dict[self.data_keys[key_index]]["lookup"]})
                df.to_csv(self.dataset + "/csv_files/" + self.data_keys[key_index] + "_lookup.csv", index_label="index",
                          sep="\t")

    def make_key_dict(self):
        """
        the key_dict contains all usefull information such as key, lookup, values, col_keys, prep_type
        :return: key_dict
        """
        key_dict = {}
        df = pd.read_csv(self.dataset + "/csv_files/" + self.file_name + ".csv", sep="\t")[self.data_keys]
        N = df.shape[0]
        self.dataset_info["n_rows"] = N

        for data_key, prep_type in zip(self.data_keys, self.prep_types):
            data = df[data_key].values
            if prep_type == "pass":
                values = data.reshape((N, 1))
                col_keys = [data_key]

            elif prep_type == "one_hot":
                values = prep.unique_int_to_one_Hot(prep.string_to_unique_int(data)).T
                col_keys = [data_key + "_" + str(u) for u in np.unique(data)]

            elif prep_type == "indexing":
                values = prep.string_to_unique_int(data).reshape((N, 1))
                col_keys = [data_key + "_index"]

            elif prep_type == "normalize":
                values = ((data - np.mean(data)) / np.var(data)).reshape((N, 1))
                col_keys = [data_key + "_normalized"]

            key_dict[data_key] = {"prep_type": prep_type,
                                  "lookup": np.unique(data),
                                  "values": values,
                                  "col_keys": col_keys}

            self.dataset_info["keys"][data_key]["n_unique"] = len(key_dict[data_key]["lookup"])
            self.dataset_info["keys"][data_key]["col_keys"] = col_keys

        if self.save_dataset_info:
            with open(self.dataset+"/json_files/dataset_info.json",'w') as fp:
                json.dump(self.dataset_info, fp, indent = 5)

        return key_dict

    def make_col_dict(self):
        """
        the col diczt has the struckture of a table
        it uses the key ddict for initialization
        :return: col_dict
        """
        col_dict = {}
        for data_key in self.data_keys:
            col_keys = self.key_dict[data_key]["col_keys"]
            for col_key_index in range(len(col_keys)):
                col_dict[col_keys[col_key_index]] = self.key_dict[data_key]["values"][:, col_key_index]
        return col_dict


    def match_clients(self, only_bool=False, client_col_name="Kundenname", client_lookup_file="Kundenname_lookup"):
        """
        merges the information about many clients and produkts to an average info vector for every client
        :param only_bool:
        :param client_col_name:
        :param client_lookup_file: for matching
        :return: client_info -> info vektor for every client
        """
        client_data = pd.DataFrame(self.col_dict).values
        client_lookup = pd.read_csv(self.dataset + "/csv_files/" + client_lookup_file + ".csv",
                                    sep="\t").values  # self.key_dict[client_col_name]["lookup"]
        n_Kunden = len(client_lookup)
        n_Aufträge, n_features = np.shape(client_data)
        client_info = np.zeros((n_Kunden, n_features - 1))
        count = np.zeros(n_Kunden)

        for data in client_data:
            index = np.argwhere(client_lookup[:, 1] == data[0])[0, 0]
            count[index] += 1
            client_info[index] = client_info[index] + np.delete(data, 0)

        for i in range(n_Kunden):
            client_info[i] = client_info[i] / count[i] if count[i] != 0 else client_info[i]

        return client_info


    def delete_col_dict_duplicates(self, dtypes=[int, str]):
        keys = self.col_dict.keys()
        values = np.array(delete_duplicates(pd.DataFrame(self.col_dict).values))
        self.col_dict = {key: value for key, value in zip(keys, values.T)}


    def col_dict_to_csv(self, title, index_label="index", sep="\t"):
        """
        saves the col_dict as csv file
        :param title: title of the csv file
        :param index_label: row index name
        :param sep: seperator of the file
        :return:
        """
        pd.DataFrame(self.col_dict).to_csv(self.dataset + "/csv_files/" + title + ".csv", index_label=index_label, sep=sep)


def get_info(dataset="Superstore",
             file_name="Superstore",
             data_keys=["Kundenname", "Segment", "Kategorie"],
             prep_types=["pass", "one_hot", "one_hot"],
             save_info=False):

    ir = information_retriever(dataset=dataset, file_name=file_name, data_keys=data_keys, prep_types=prep_types)
    client_info = ir.match_clients(client_col_name=data_keys[0], client_lookup_file=data_keys[0] + "_lookup")
    if save_info:
        df = pd.DataFrame({list(ir.col_dict.keys())[index + 1]: client_info[:, index] for index in
                           range(len(list(ir.col_dict.keys())) - 1)})
        df.to_csv(ir.dataset + "/csv_files/" + file_name + "_info.csv", index_label="Client", sep="\t")
    del ir
    return client_info


def get_orders(dataset="Superstore",
               file_name="Superstore",
               data_keys=["Kundenname", "Produktname"],
               prep_types=["indexing", "indexing"],
               save_info=False):
    ir = information_retriever(dataset=dataset, file_name=file_name, data_keys=data_keys, prep_types=prep_types)
    if save_info:
        pd.DataFrame(ir.col_dict).to_csv(dataset + "/csv_files/" + dest_file + ".csv", index_label="Auftrag_nr",
                                         sep="\t")
    values = pd.DataFrame(ir.col_dict).values
    del ir
    return values


def get_lookups(dataset="Superstore",
                file_name="Superstore",
                data_keys=["Kundenname", "Produktname"],
                prep_types=["indexing", "indexing"],
                save_info=False):
    ir = information_retriever(dataset=dataset, file_name=file_name, data_keys=data_keys, prep_types=prep_types,
                               save_lookups=True)
    del ir


if __name__ == "__main__":
    dataset = sys.argv[1]
    file_name = sys.argv[2]
    dest_file = sys.argv[3]
    args = sys.argv[4:]
    data_keys, prep_types = [], []
    l = data_keys
    for arg in args:
        if arg == "-data_keys":
            l = data_keys
        elif arg == "-prep_types":
            l = prep_types
        else:
            l.append(arg)
    ir = information_retriever(dataset=dataset, file_name=file_name, data_keys=data_keys, prep_types=prep_types,
                               save_lookups=True)
    pd.DataFrame(ir.col_dict).to_csv(dataset + "/csv_files/" + dest_file + ".csv", index_label="Auftrag_nr", sep="\t")

