import json
import os
import sys
from tools.data_splitter import split_data
from tools import change_seperator
import shutil
import pandas as pd

def make_config_file(dataset):
    '''
    Initialize a config file (save at directory: dataset/json_files/config.json) which consists the the "dataset":dataset und "file_name_list":[dataset,dataset_train,dataset_test,dataset_valid]
    a csv file containing the neccessary information named like csv_filename = dataset_filename should be placed at directory dataset_name/csv_files/dataset_name.csv
    the col names of te relevant columnes are missing. fill in the parameter col_name_list as a list of strings Bsp. ["Kundenname","Produktname"]
    :param dataset: name_of_the_dataset
    :return: config.json
    '''
    file_name_list = []
    for type in ["", "_train", "_test", "_valid"]:
        file_name_list.append(dataset+type)

    config =  {
                 "dataset": dataset,
                 "file_name_list": file_name_list,
                 "col_name_list": [
                      "Kundenname",
                      "Produktname"
                 ],
                 "show_progress": True,
                 "use_user_info": False,
                 "user_features": [],
                 "user_features_prep_types": [],
                 "model":"DeepLearning",
                 "model_name":dataset+"_model",
                 "DeepLearning":{
                    "weights_name":dataset+"_weights",
                    "n_epochs":1
                 },
                 "LuwiNet":{
                    "n_train_clients": 1,
                    "n_contents": 1,
                    "learning_rate": 0.005,
                    "n_epochs": 1000,
                    "embeddding_size": 1024,
                    "hidden_client_dims": [],
                    "hidden_content_dims": []
                 },
                 "NaiveBayes":{
                    "alpha": 1,
                    "fit_prior": True,
                    "model_type": "multinomial"
                 },

                 "approach": "multi",
                 "count": False,
                 "split_ratio": [0.7,0.2,0.1],
                 "split": "clients",
                 "train_batch_size": 10000,
                 "pred_batch_size": 10000,
                 "n_train_batches": 0,
                 "n_pred_batches": 0,
                 "n_Produkte": 0,
                 "n_Kunden": 0,
                 "info_string": "",
                 "fit_set": "train",
                 "pred_set": "test",
                 "n_info_cols": 0,
                 "save_lookups_to_db":False,
                 "save_result_to_db":False,
                 "save_dataset_to_db":False,
                 "save_as_csv":False
            }

    with open(dataset + "/json_files/config.json", 'w') as fp:
        json.dump(config, fp, indent=5)

    return config

def config_ui(config,possible_keys):
    possible_keys = list(possible_keys)

    print("Set up the configurations for Naive Bayes \n")
    print("possible keys in dataset:")
    for key in possible_keys:
        print(key)


    valid = False
    while not valid:
        input_=input("whats the name of the client name/index column? (check for misspelling!)\n"
                            "Input: ")

        if input_ in possible_keys:
            client_col_name = input_
            valid = True
        else:
            print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_=input("whats the name of the product name/index column? (check for misspelling!)\n"
                             "Input: ")

        if input_ in possible_keys and not (input_ == client_col_name):
            product_col_name = input_
            valid = True
        else:
            print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_=input(   "Do you want to use additional features  \n"
                        "(for example country or age... could be any column in your dataset)?  \n"
                        "type 1 if you want to use additional features, type 0 if not \n"
                        "Input: ")

        if input_ == "":
            input_ = "0"
            print("default value set: " + input_)

        if input_ in ["1","0"]:
            use_user_info = bool(int(input_))
            valid = True
        else:
            print("No valid input. Repeat! \n")

    user_features = []
    user_features_prep_types = []

    if use_user_info:
        stop = False
        i = 0
        while not stop:
            i += 1
            print("which columns contain the information u want to use")
            print("possible keys in dataset:")
            for key in possible_keys:
                print(key)


            valid = False
            while not valid:
                input_ = input("type the name of the " + str(i) +
                            " user_info column. type 'stop' if there are no more user features \n"
                            "Input: ")

                if input_ == "":
                    input_ = "stop"
                if input_ == "stop":
                    feature= "stop"
                    break

                if input_ in possible_keys and not (input_ in [client_col_name,product_col_name]+user_features):
                    feature = input_
                    valid = True
                else:
                    print("No valid input. Repeat! \n")

            if feature == "":
                feature = "stop"

            if feature == "stop":
                break
            user_features.append(feature)


            valid = False
            while not valid:
                input_ = input( "how you want to preprocess this information(" + feature + ")?  \n"
                                "choices: \n "
                                "'pass' for (bool feature) \n "
                                "'one_hot' for (multiclass feature) \n "
                                "'normalize' for (positive real number feature)] \n"
                                "Input: ")

                if input_ == "":
                    input_ = "one_hot"
                    print("default value set: " + input_)

                if input_ in ["pass","one_hot","normalize"]:
                    user_features_prep_types.append(input_)
                    valid=True
                else:
                    print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_=input("\nDo you want to print out the progress of computation steps?\n"
                                   "type 1 if you want it, type 0 if not \n"
                                   "Input: ")

        if input_ == "":
            input_ = "1"
            print("default value set: " + input_)

        if input_ in ["1","0"]:
            show_progress = bool(int(input_))
            valid = True
        else:
            print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_=input("\nDo you want to save the source dataset to the Database?\n"
                                   "type 1 if you want it, type 0 if not \n"
                                   "Input: ")

        if input_ == "":
            input_ = "0"
            print("default value set: " + input_)

        if input_ in ["1","0"]:
            save_dataset_to_db  = bool(int(input_))
            valid = True
        else:
            print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_=input("\nDo you want to save the lookups to the Database?\n"
                                   "type 1 if you want it, type 0 if not \n"
                                   "Input: ")

        if input_ == "":
            input_ = "0"
            print("default value set: " + input_)

        if input_ in ["1","0"]:
            save_lookups_to_db = bool(int(input_))
            valid = True
        else:
            print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_=input("\nDo you want to save the result to the Database?\n"
                                   "type 1 if you want it, type 0 if not \n"
                                   "Input: ")

        if input_ == "":
            input_ = "0"
            print("default value set: " + input_)

        if input_ in ["1","0"]:
            save_result_to_db = bool(int(input_))
            valid = True
        else:
            print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_=input("\nDo you want to save the results as csv?\n"
                                   "type 1 if you want it, type 0 if not \n"
                                   "Input: ")

        if input_ == "":
            input_ = "0"
            print("default value set: " + input_)

        if input_ in ["1","0"]:
            save_as_csv = bool(int(input_))
            valid = True
        else:
            print("No valid input. Repeat! \n")


    valid = False
    while not valid:
        input_ = input("\ndo you want to go deeper into the configuration? otherwise default configurations will be set? \n"
                      "type 1 if you want it, type 0 if not \n"
                      "Input: ")

        if input_ == "":
            input_ = "0"
            print("default value set: " + input_)

        if input_ in ["1", "0"]:
            valid = True
        else:
            print("No valid input. Repeat! \n")

    if bool(int(input_)):
        valid = False
        while not valid:
            input_ = input("\nWhich Classification Algorithm do you want to use?\n "
                                   "type DeepLearning for DeepLearning\ntype NaiveBayes for NaiveBayes\ntype LuwiNet for LuwiNet\t"
                                   "Input: ")

            if input_ == "":
                input_ = "NaiveBayes"
                print("default value set: " + input_)

            if input_ in ["DeepLearning", "NaiveBayes","LuwiNet"]:
                valid = True
                model = input_
            else:
                print("No valid input. Repeat! \n")


        if model == "DeepLearning":
            model_type = "multinomial"
            fit_prior = True
            alpha = 1

            valid = False
            while not valid:
                input_ = input("\nhow many epoch u want to train\n"
                               "Input (int): ")

                if input_ == "":
                    input_ = 1
                    print("default value set: " + input_)

                try:
                    n_epochs = int(input_)
                    valid = True
                except:
                    print("No valid input. Repeat! \n")

        elif model == "NaiveBayes":
            n_epochs = 1

            valid = False
            while not valid:
                input_ = input("\nWhich Naive Bayes model do you want to use \n"
                               "Type 'multinomial' for Sklearn MultinomialNB \n"
                               "Type 'complement' for Sklearn ComplementNB \n"
                               "Type 'bernoulli' for Sklearn BernoulliNB \n"
                               "Type 'gaussian' for Sklearn GaussianNB \n"
                               "Input: ")

                if input_ == "":
                    input_ = "multinomial"
                    print("default value set: " + input_)

                if input_ in ["multinomial", "complement", "bernoulli", "gaussian"]:
                    valid = True
                    model_type = input_
                else:
                    print("No valid input. Repeat! \n")


            valid = False
            while not valid:
                input_ = input("\nDo you want to calculate the prior distribution from the Databasis?\n "
                                       "type 1 if you want it, type 0 if not \n"
                                       "Input: ")

                if input_ == "":
                    input_ = "1"
                    print("default value set: " + input_)

                if input_ in ["1", "0"]:
                    valid = True
                    fit_prior = bool(int(input_))
                else:
                    print("No valid input. Repeat! \n")

            valid = False
            while not valid:
                input_ = input("\nDo you want to set a smoothing parameter alpha in intervall [0|1].\n "
                                    "Type the value default is 1, make shure its a float value\n"
                                    "Input: ")

                if input_ == "":
                    input_ = "1"
                    print("default value set: " + input_)

                try:
                    alpha = float(input_)
                    if alpha > 0:
                        valid = True
                except:
                    print("No valid input. Repeat! \n")

        elif model == "LuwiNet":

            valid = False
            while not valid:
                input_ = input("\nWhat learning_rate would you prefer [0|1].\n "
                                    "Type the value default is 1, make shure its a float value\n"
                                    "Input: ")

                if input_ == "":
                    input_ = "1"
                    print("default value set: " + input_)

                try:
                    learning_rate = float(input_)
                    if learning_rate > 0:
                        valid = True
                except:
                    print("No valid input. Repeat! \n")

            valid = False
            while not valid:
                input_ = input("\nHow many epochs would you like to train?\n "
                                    "Type the value default is 1, make shure its a int value\n"
                                    "Input: ")

                if input_ == "":
                    input_ = "1"
                    print("default value set: " + input_)

                try:
                    n_epochs = int(input_)
                    if n_epochs > 0:
                        valid = True
                except:
                    print("No valid input. Repeat! \n")

            valid = False
            while not valid:
                input_ = input("\nWhat size would you like to embed your vectors in?\n "
                               "Type the value default is 100, make shure its a int value\n"
                               "Input: ")

                if input_ == "":
                    input_ = "100"
                    print("default value set: " + input_)

                try:
                    embeddding_size = int(input_)
                    if embeddding_size > 0:
                        valid = True
                except:
                    print("No valid input. Repeat! \n")

            valid = False
            while not valid:
                input_ = input("\nDo you want to design a personal Network Architecture?\n "
                               "Type the 1 for Yes and 0 for no default is 0, leads to logistic regression vector embedding \n"
                               "Input: ")

                if input_ == "":
                    input_ = "0"
                    print("default value set: " + input_)

                try:
                    net_architecture = int(input_)
                    valid = True
                except:
                    print("No valid input. Repeat! \n")

            if net_architecture:
                hidden_client_dims = []
                stop = False
                i = 0
                while not stop:
                    i += 1
                    print("what should be the size of the first "+i+"'s hidden layer of the client embedding network")

                    valid = False
                    while not valid:
                        input_ = input("what should be the size of the first "+i+"'s hidden layer."+"type 'stop' if there are no more layers \n"
                                        "Input: ")

                        if input_ == "":
                            input_ = "stop"
                        if input_ == "stop":
                            feature = "stop"
                            break

                        try:
                            dim = int(input_)
                            valid = True
                        except:
                            print("No valid input. Repeat! \n")

                    if feature == "":
                        feature = "stop"

                    if feature == "stop":
                        break
                    hidden_client_dims.append(dim)

                hidden_content_dims = []
                stop = False
                i = 0
                while not stop:
                    i += 1
                    print(
                        "what should be the size of the first " + i + "'s hidden layer of the content embedding network")

                    valid = False
                    while not valid:
                        input_ = input(
                            "what should be the size of the first " + i + "'s hidden layer." + "type 'stop' if there are no more layers \n"
                                                                                               "Input: ")

                        if input_ == "":
                            input_ = "stop"
                        if input_ == "stop":
                            feature = "stop"
                            break

                        try:
                            dim = int(input_)
                            valid = True
                        except:
                            print("No valid input. Repeat! \n")

                    if feature == "":
                        feature = "stop"

                    if feature == "stop":
                        break
                    hidden_content_dims.append(dim)
            else:
                learning_rate = 0.005
                n_epochs = 1000
                embeddding_size = 100
                hidden_client_dims = []
                hidden_content_dims = []


        valid = False
        while not valid:
            input_ = input("\nWhich classification approach do you want to use?  \n"
                         "Type 'binary' for binary classification (slow but good performance) \n"
                         "Type 'multi' for multiclass classification (fast but maybe worse performance) \n"
                         "Input: ")

            if input_ == "":
                input_ = "multi"
                print("default value set: " + input_)

            if input_ in ["binary", "multi"]:
                valid = True
                approach = input_
            else:
                print("No valid input. Repeat! \n")


        valid = False
        while not valid:
            input_ = input("\nHow do you want to split your Dataset? \n"
                      "Type 'clients' for a split along clients  \n"
                      "Type 'orders' for a split along orders  \n"
                      "Input: ")

            if input_ == "":
                input_ = "clients"
                print("default value set: " + input_)

            if input_ in ["clients", "orders"]:
                valid = True
                split = input_
            else:
                print("No valid input. Repeat! \n")


        valid = False
        while not valid:
            input_ = input(
                "\ndo you want to go even deeper into the configuration? otherwise default configurations will be set? \n"
                "type 1 if you want it, type 0 if not \n"
                "Input: ")

            if input_ == "":
                input_ = "0"
                print("default value set: " + input_)

            if input_ in ["1", "0"]:
                valid = True
            else:
                print("No valid input. Repeat! \n")

        if bool(int(input_)):
            valid = False
            while not valid:
                input_ = input("\nDo you want to count the amount of products that people buy?\n "
                               "type 1 if you want it, type 0 if not \n"
                               "Input: ")

                if input_ == "":
                    input_ = "0"
                    print("default value set: " + input_)

                if input_ in ["1", "0"]:
                    valid = True
                    count = bool(int(input_))
                else:
                    print("No valid input. Repeat! \n")


            valid = False
            while not valid:
                try:
                    train_split = float(input("\nWhich part of the Dataset should be the Trainingset. \n"
                                              "Type the value like (0.7) in the intervall [0|1], make shure its a float value\n"
                                              "Input: "))
                    if train_split <= 1 and train_split >= 0:
                        valid = True
                    else:
                        print("No valid input. Repeat! value should be in Intervall [0|1]\n")
                except:
                    print("No valid input. Repeat! \n")


            valid = False
            while not valid:
                try:
                    test_split = float(input("\nWhich part of the Dataset should be the Testset. \n"
                                             "Type the value like (0.2) in the intervall [0|1], make shure its a float value \n"
                                             "Input: "))
                    if test_split <= 1 and test_split >= 0:
                        valid = True
                    else:
                        print("No valid input. Repeat! value should be in Intervall [0|1]\n")
                except:
                    print("No valid input. Repeat! \n")


            valid = False
            while not valid:
                try:
                    valid_split = float(input("\nWhich part of the Dataset should be the Validset. \n"
                                              "Type the value like (0.1) in the intervall [0|1], make shure its a float value\n"
                                              "Input: "))
                    if valid_split <= 1 and valid_split >= 0:
                        valid = True
                    else:
                        print("No valid input. Repeat! value should be in Intervall [0|1]\n")
                except:
                    print("No valid input. Repeat! \n")

            split_ratio = [train_split, test_split, valid_split]


            valid = False
            while not valid:
                input_ = input("\nWhich set do you want to train on?\n"
                                " Type the name (for example 'train', choices ['full','train','test','valid'])\n"
                                "Input: ")

                if input_ == "":
                    input_ = "train"
                    print("default value set: " + input_)

                if input_ in ["full", "train", "test", "valid"]:
                    valid = True
                    fit_set = input_
                else:
                    print("No valid input. Repeat! \n")


            valid = False
            while not valid:
                input_ = input("\nWhich set do you want to predict on?\n"
                                " Type the name (for example 'train', choices ['full','train','test','valid'])\n"
                                "Input: ")

                if input_ == "":
                    input_ = "test"
                    print("default value set: " + input_)

                if input_ in ["full", "train", "test", "valid"]:
                    valid = True
                    pred_set = input_
                else:
                    print("No valid input. Repeat! \n")


            valid = False
            while not valid:
                try:
                    train_batch_size = int(
                        input("\nWhat should the batch size of the training data? needed if you get a Memory error"
                              ", make shure its a integer value\n"
                              "Input: "))
                    valid = True
                except:
                    print("No valid input. Repeat! \n")


            valid = False
            while not valid:
                try:
                    pred_batch_size = int(
                        input("\nWhat should the batch size of the predicted data? needed if you get a Memory error"
                              ", make shure its a integer value\n"
                              "Input: "))
                    valid = True
                except:
                    print("No valid input. Repeat! \n")

        else:
            split_ratio = [0.7, 0.2, 0.1]
            count = False
            fit_set = "train"
            pred_set = "test"
            train_batch_size = 10000
            pred_batch_size = 10000
    else:
        fit_prior = True
        count = False
        model="NaiveBayes"
        alpha = 1
        n_epochs = 1
        approach = "multi"
        model_type = "multinomial"
        split = "orders"
        split_ratio = [0.7, 0.2, 0.1]
        fit_set = "train"
        pred_set = "test"
        train_batch_size = 10000
        pred_batch_size = 10000

    print("client_col_name", client_col_name)
    print("product_col_name", product_col_name)
    print("use_user_info", use_user_info)
    print("user_features", user_features)
    print("user_features_prep_types", user_features_prep_types)
    print("show_progress", show_progress)
    print("model", model)
    if model == "DeepLearning":
        print()
    elif model == "NaiveBayes":
        print("model_type", model_type)
        print("alpha", alpha)
        print("fit_prior",fit_prior)
    elif model == "LuwiNet":
        print("learning_rate",learning_rate)
        print("n_epochs",n_epochs)
        print("embeddding_size",embeddding_size)
        print("hidden_content_dims",hidden_content_dims)
        print("hidden_client_dims",hidden_client_dims)

    print("count", count)
    print("approach",approach)
    print("split", split)
    print("split_ratio", split_ratio)
    print("fit_set", fit_set)
    print("pred_set", pred_set)
    print("train_batch_size", train_batch_size)
    print("pred_batch_size", pred_batch_size)
    print("save_as_csv",save_as_csv)
    print("save_to_db",save_dataset_to_db)
    print("save_to_db", save_result_to_db)
    print("save_to_db", save_lookups_to_db)

    config["learning_rate"] = learning_rate
    config["n_epochs"] = n_epochs
    config["embeddding_size"] = embeddding_size
    config["hidden_content_dims"] = hidden_content_dims
    config["hidden_client_dims"] = hidden_client_dims
    config["col_name_list"] = [client_col_name,product_col_name]
    config["use_user_info"] = use_user_info
    config["user_features"] = user_features
    config["user_features_prep_types"] = user_features_prep_types
    config["show_progress"] = show_progress
    config["NaiveBayes"]["fit_prior"] = fit_prior
    config["DeepLearning"]["n_epochs"] = n_epochs
    config["model"]=model
    config["count"] = count
    config["NaiveBayes"]["alpha"] = alpha
    config["NaiveBayes"]["model_type"] = model_type
    config["split"] = split
    config["split_ratio"] = split_ratio
    config["fit_set"] = fit_set
    config["pred_set"] = pred_set
    config["train_batch_size"] = train_batch_size
    config["pred_batch_size"] = pred_batch_size
    config["approach"] = approach
    config["save_dataset_to_db"] = save_dataset_to_db
    config["save_lookups_to_db"] = save_lookups_to_db
    config["save_result_to_db"] = save_result_to_db
    config["save_result_as_csv"] = save_as_csv


    return config

def do(dataset):
    '''
    - if necessary create dataset directory
    - if necessary create subdirectories: "npy_files","csv_files","batches","plots","json_files","models"
    - check if the dataser csv file is present
    - if necessary create an initialized config.json file
    - if neccessary split the dataset into subsets: full, train, test, valid

    :param dataset:
    :return:
    '''

    if not os.path.isdir(dataset):
        os.mkdir(dataset)

    for subdir in ["npy_files", "csv_files", "batches", "plots", "json_files", "models"]:
        if not os.path.isdir(dataset+"/"+subdir):
            os.mkdir(dataset+"/"+subdir)

    for subdir in ["DeepLearning","NaiveBayes"]:
        if not os.path.isdir(dataset+"/models/"+subdir):
            os.mkdir(dataset+"/models/"+subdir)

    for subdir in ["weights","models"]:
        if not os.path.isdir(dataset+"/models/DeepLearning/"+subdir):
            os.mkdir(dataset+"/models/DeepLearning/"+subdir)

    if not os.path.isfile(dataset+"/csv_files/"+dataset+".csv"):
        print("no csv file found! \n")
        file_path = input("type the absolut path to your csv file! \nInput:")
        shutil.copyfile(file_path, dataset+"/csv_files/"+dataset+".csv")

    possible_keys = pd.read_csv(dataset+"/csv_files/"+dataset+".csv",sep = "\t").keys()
    if len(possible_keys) == 1:
        print(possible_keys[0])
        current_sep = input("Only one column recognized! The seperator has to be changed! \nWhats the current Seperator? \nInput:")
        change_seperator.do(dataset,dataset,current_sep)
        possible_keys = pd.read_csv(dataset + "/csv_files/" + dataset + ".csv", sep="\t").keys()

    if not os.path.isfile(dataset + "/json_files/config.json"):
        config = make_config_file(dataset)
        print(os.path.isfile(dataset + "/json_files/config.json"))
    else:
        with open(dataset + "/json_files/config.json", 'r') as fp:
            config = json.load(fp)

    config = config_ui(config,possible_keys)

    if not all([os.path.isfile(dataset+"csv_files/"+dataset+dset+".csv") for dset in ["", "_train", "_test", "_valid"]]):
        split_data(dataset, dataset, config["split_ratio"], config["col_name_list"]+config["user_features"])

    with open(dataset + "/json_files/config.json", 'w') as fp:
        json.dump(config, fp, indent=5)


if __name__ == "__main__":
    dataset = sys.argv[1]
    do(dataset)
