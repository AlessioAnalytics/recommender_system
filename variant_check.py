import serve_data
import NaiveBayes
import configure
import Tableau_export
from sklearn import metrics
import json
import pandas as pd
import numpy as np


def update_config(dataset, approach, model_type, split, use_user_info, user_features, user_features_prep_types, count, fit_prior , alpha):

    with open(dataset + "/json_files/config.json", "r") as fp:
        config = json.load(fp)
    config["approach"] = approach
    config["model_type"] = model_type
    config["split"] = split
    config["use_user_info"] = use_user_info
    config["user_features"] = user_features
    config["user_features_prep_types"] = user_features_prep_types
    config["count"] = count
    config["fit_prior"] = fit_prior
    config["alpha"] = alpha

    with open(config["dataset" ] +"/json_files/" +"config.json", 'w') as fp:
        json.dump(config, fp, indent=5)


def check_all_variants( start_config=   {
                                                "dataset": "Superstore",
                                                "file_name_list": [
                                                    "Superstore",
                                                    "Superstore_train",
                                                    "Superstore_test",
                                                    "Superstore_valid"
                                                ],
                                                "col_name_list": [
                                                    "Kundenname",
                                                    "Produktname"
                                                ],
                                                "show_progress": False,
                                                "use_user_info": False,
                                                "user_features": [
                                                    "Segment",
                                                    "Kategorie"
                                                ],
                                                "user_features_prep_types": [
                                                    "one_hot",
                                                    "one_hot"
                                                ],
                                                "n_info_cols": 0,
                                                "approach": "binary",
                                                "model_type": "complement",
                                                "count": True,
                                                "split_ratio": [
                                                          0.7,
                                                          0.2,
                                                          0.1
                                                     ],
                                                "split": "clients",
                                                "alpha": 1.0,
                                                "fit_prior": True,
                                                "train_batch_size": 5000,
                                                "pred_batch_size": 5000,
                                                "n_Produkte": 1915,
                                                "n_Kunden": 784,
                                                "info_string": "",
                                                "fit_set": "train",
                                                "pred_set": "test",
                                        },
                        param_dict =    {
                                                "approach": ["multi", "binary"],
                                                "model_type": ["multinomial", "complement", "bernoulli"],
                                                "split": ["clients", "orders"],
                                                "use_user_info": [False, True],
                                                "count": [True, False],
                                                "alpha": [1.0,0.9,0.8,0.7],
                                                "fit_prior": [True,False],
                                                "user_features": [["Segment"], ["Kategorie"], ["Segment", "Kategorie"]],
                                                "user_features_prep_types": [["one_hot"], ["one_hot"], ["one_hot", "one_hot"]]
                                        }):
    top_n_list = [10, 20, 50, 100, 200, 500]

    full_out_dict = {
        "approach": [],
        "model_type":[],
        "split": [],
        "use_user_info": [],
        "threshold": [],
        "count": [],
        "info_str": [],
        "filename": [],
        "mse": [],
        "neg_log_loss": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "tn": [],
        "fp": [],
        "fn": [],
        "tp": []
    }


    for top_n in top_n_list:
        full_out_dict["top_" + str(top_n) + "_score"] = []

    dataset = start_config["dataset"]
    #configure.do(dataset)

    with open(dataset + "/json_files/config.json", "w") as fp:
        json.dump(start_config, fp, indent=5)

    for approach in param_dict["approach"]:
        for model_type in param_dict["model_type"]:
            for split in param_dict["split"]:
                for count in param_dict["count"]:
                    for fit_prior in param_dict["fit_prior"]:
                        for alpha in param_dict["alpha"]:
                            for use_user_info in param_dict["use_user_info"]:
                                for user_features ,user_features_prep_types in zip(param_dict["user_features"],param_dict["user_features_prep_types"]):

                                    if not use_user_info and not user_features == param_dict["user_features"][-1]:
                                        continue

                                    update_config(dataset,
                                                  approach,
                                                  model_type,
                                                  split,
                                                  use_user_info,
                                                  user_features,
                                                  user_features_prep_types,
                                                  count,
                                                  fit_prior,
                                                  alpha)

                                    if use_user_info:
                                        info_str= str(user_features)
                                    else:
                                        info_str = ""
                                    print()
                                    print("Precess with new config:")
                                    print("approach", "model_type", "split", "use_user_info", "info_str", "count", "fit_prior", "alpha")
                                    print(approach, model_type, split, use_user_info, info_str, count, fit_prior, alpha)
                                    print()

                                    serve_data.do(dataset)
                                    NaiveBayes.do(dataset)

                                    with open(dataset + "/json_files/config.json", "r") as fp:
                                        config = json.load(fp)

                                    title = dataset + "_predictions_" + \
                                            "fit" + config["fit_set"] + \
                                            "_pred" + config["pred_set"] + \
                                            "_" + config["model_type"] + \
                                            "_approach" + str(config["approach"]) + \
                                            "_split" + config["split"] + \
                                            "_count" + str(config["count"]) + \
                                            "_info" + str(config["use_user_info"]) + config["info_string"]

                                    pred_file = dataset + "/npy_files/" + title + ".npy"

                                    if split == "orders":
                                        KPM = np.sign(np.load(dataset+"/npy_files/test_KPM.npy"))
                                    elif split == "clients":
                                        KPM = np.sign(np.load(dataset+"/npy_files/full_KPM.npy")
                                                      [np.load(dataset+"/npy_files/test_index.npy")])

                                    if approach == "binary":
                                        threshold = 0.5
                                    elif approach == "multi":
                                        threshold = 1/config["n_Produkte"]

                                    n_orders = np.sum(KPM, axis=None)
                                    predictions = np.load(pred_file)
                                    y_prop = predictions.flatten()
                                    y_soll = KPM.flatten()
                                    y_pred = y_prop > threshold

                                    top_n_score_list = []
                                    for top_n in top_n_list:
                                        n_hits = 0
                                        for client_index in range(len(predictions)):
                                            bought_items = np.argwhere(KPM[client_index] == 1)[:, 0]
                                            for item_index in bought_items:
                                                if item_index in np.array(
                                                        sorted(zip(predictions[client_index], np.arange(len(
                                                            predictions[client_index]))), reverse=True))[:, 1][:top_n]:
                                                    n_hits += 1
                                        top_n_score_list.append(n_hits / n_orders)
                                    cmat = metrics.confusion_matrix(y_soll, y_pred)
                                    [[tn, fp], [fn, tp]] = cmat
                                    out_dict = {
                                        "filename": str(pred_file),
                                        "mse": float(metrics.mean_squared_error(y_soll, y_prop)),
                                        "neg_log_loss": float(metrics.log_loss(y_soll, y_prop)),
                                        "Accuracy": float(metrics.accuracy_score(y_soll, y_pred)),
                                        "Precision": float(metrics.precision_score(y_soll, y_pred)),
                                        "Recall": float(metrics.recall_score(y_soll, y_pred)),
                                        "F1": float(metrics.f1_score(y_soll, y_pred)),
                                        "tn": int(tn),
                                        "fp": int(fp),
                                        "fn": int(fn),
                                        "tp": int(tp)
                                    }

                                    for top_n, score in zip(top_n_list, top_n_score_list):
                                        full_out_dict["top_" + str(top_n) + "_score"].append(float(score))

                                    print(pred_file + ":")
                                    print("MSE", out_dict["mse"])
                                    print("neg_log_loss", out_dict["neg_log_loss"])
                                    print("Accuracy", out_dict["Accuracy"])
                                    print("Precision", out_dict["Precision"])
                                    print("Recall", out_dict["Recall"])
                                    print("F1", out_dict["F1"])

                                    print("Confusion Matrix (tn,fp,fn,tp)")
                                    print(cmat)

                                    for top_n ,score in zip(top_n_list, top_n_score_list):
                                        print(str(score * 100) + "%\tder getätigten käufte sind in der top",
                                              top_n, "der Produktempfehlungen")

                                    full_out_dict["filename"].append(str(pred_file))
                                    full_out_dict["approach"].append(str(approach))
                                    full_out_dict["model_type"].append(str(model_type))
                                    full_out_dict["split"].append(str(split))
                                    full_out_dict["count"].append(str(count))
                                    full_out_dict["use_user_info"].append(str(use_user_info))
                                    full_out_dict["threshold"].append(float(threshold))
                                    full_out_dict["info_str"].append(str(info_str))
                                    full_out_dict["fit_prior"].append(fit_prior)
                                    full_out_dict["alpha"].append(float(alpha))
                                    full_out_dict["mse"].append(float(out_dict["mse"]))
                                    full_out_dict["neg_log_loss"].append(float(out_dict["neg_log_loss"]))
                                    full_out_dict["Accuracy"].append(float(out_dict["Accuracy"]))
                                    full_out_dict["Precision"].append(float(out_dict["Precision"]))
                                    full_out_dict["Recall"].append(float(out_dict["Recall"]))
                                    full_out_dict["F1"].append(float(out_dict["F1"]))
                                    full_out_dict["tn"].append(int(out_dict["tn"]))
                                    full_out_dict["fp"].append(int(out_dict["fp"]))
                                    full_out_dict["fn"].append(int(out_dict["fn"]))
                                    full_out_dict["tp"].append(int(out_dict["tp"]))

                                    pd.DataFrame(full_out_dict).to_csv(dataset+"/csv_files/variant_check.csv",
                                                                       index_label="row_index", sep=";")
                                    print("-"*100)


if __name__ == "__main__":
    check_all_variants()
