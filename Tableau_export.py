from serve_data import data_server
import json
from tools.load import loader
from tools.database_connection import insert_to_table,create_table,delete_table,table_exists
import pandas as pd
import numpy as np
import datetime
import sys

def export_as_csv_in_tableau_format(data_server,config={}):
    db_config = {
        "database": {
            "host": "192.168.178.14",
            "user": "lwidowski",
            "passwd": "$moothOperat0r",
            "database": "dsaas"
        },
        "schema_name": "dbt_recommender_system_predictions",
        "table_name": "tb_b_recommendations_"+config["model_name"],
        "key_list": ['run_id','client', 'content', 'propability', 'already_bought'],
        "dtype_list": ["int8","int","int","float","int"],
        "primary_key": None,
        "auto_increment": None
    }
    run_id = to_integer(datetime.datetime.now())
    if table_exists(db_config) == 1:
        delete_table(db_config)
    create_table(db_config)

    for batch_index in range(config["n_pred_batches"]):

        if config["show_progress"] and config["n_pred_batches"] != 1:
            if batch_index == 0:
                load = loader(config["n_pred_batches"],"export")
            load.print_progress(batch_index)

        predictions = np.load(config["dataset"] + "/batches/prediction_batch_no_" + str(batch_index) + ".npy")
        KPM = np.load(config["dataset"] + "/batches/KPM_batch_no_" + str(batch_index) + ".npy")
        indexes = np.load(config["dataset"] + "/batches/indexes_batch_no_" + str(batch_index) + ".npy")
        if len(predictions) < config["pred_batch_size"] and  batch_index != config["n_pred_batches"]-1:
            print(  """
                    prediction batch size ("pred_batch_size" = %d) is too high.
                    change value in config file.
                    For now it has been set to the length of the prediction.
                    """ % config["pred_batch_size"])
            config["pred_batch_size"] = len(predictions)

        n_Kunden, n_Produkte = predictions.shape
        dict = {"client": [], "content": [], "propability": [], "already_bought": []}
        for k in range(n_Kunden):

            if config["show_progress"] and config["n_pred_batches"] == 1:
                if k == 0:
                    load=loader(n_Kunden, "export batch: "+str(batch_index+1)+" von"+str(config["n_pred_batches"]))
                load.print_progress(k)

            for p in range(n_Produkte):
                if config["split"] == "clients":
                    dict["client"].append(indexes[k])
                    dict["already_bought"].append(KPM[k, p])
                else:
                    dict["client"].append(k)
                    dict["already_bought"].append(KPM[k, p])
                dict["content"].append(p)
                dict["propability"].append(predictions[k, p])

        # title = config["dataset"] + "_predictions_batch_no_" + str(batch_index) + \
        #             "_fit" + config["fit_set"] + "_pred" + config["pred_set"] + "_" + \
        #             config["NaiveBayes"]["model_type"] + "_approach" + str(config["approach"]) + \
        #             "_split" + config["split"] + "_info" + str(config["use_user_info"]) + config["info_string"]
        title = config["model_name"] + "_predictions"
        df = pd.DataFrame(dict)

        if config["save_result_as_csv"]:
            df.to_csv("Tableau_exports/" + title + ".csv", index_label="Row_index", sep=";")
        if config["save_result_to_db"]:
            r,c = df.values.shape
            values = np.hstack((np.ones((r,1))*run_id,df.values))
            insert_to_table(db_config,values,config["show_progress"])

def to_integer(dt_time):
    return (10000*dt_time.year + 100*dt_time.month + dt_time.day)*1000000+10000*dt_time.hour+100*dt_time.minute++dt_time.second

def save_lookups_tb_db(config):
    for lookup_name in config["col_name_list"]:
        db_lookup_config = {
            "database": {
                "host": "192.168.178.14",
                "user": "lwidowski",
                "passwd": "$moothOperat0r",
                "database": "dsaas"
            },
            "schema_name": "dbt_recommender_system_lookups",
            "table_name": "tb_b_recommendations_" + config["model_name"] + "_" + lookup_name,
            "key_list": [ "index", lookup_name],
            "dtype_list": ["int","varchar(64)"],
            "primary_key": None,
            "auto_increment": None
        }
        vals = pd.read_csv(config["dataset"]+"/csv_files/"+lookup_name+"_lookup.csv",sep="\t").values
        if table_exists(db_lookup_config) == 1:
            delete_table(db_lookup_config)
        +create_table(db_lookup_config)
        insert_to_table(db_lookup_config,vals,config["show_progress"])


def do(dataset):
    print("load config")
    with open(dataset + "/json_files/config.json", 'r') as fp:
        config = json.load(fp)
    if config["save_result_to_db"] or config["save_result_as_csv"]:
        ds = data_server(config,calc_vals=False)
        print("start export")
        export_as_csv_in_tableau_format(ds, config)
        print("export finished")
        save_lookups_tb_db(config)
    else:
        print("export skipped")


if __name__ == "__main__":
    dataset = sys.argv[1].strip()
    do(dataset)
