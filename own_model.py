import numpy as np
import sys
from tools.load import loader
from sklearn import metrics
import matplotlib.pyplot as plt

class own_model():
    def __init__(self):
        pass

    def get_prod_occ(self,KPM,n_Kunden):
        return np.sum(KPM,axis = 0)/n_Kunden



    def fit_and_predict(self,KPM,user_info=0,method="Naive"):
        n_Kunden,n_Produkte = np.shape(KPM)
        prod_occ = self.get_prod_occ(KPM,n_Kunden)
        predictions = np.zeros_like(KPM)
        for kunden_index in range(n_Kunden):
            for kunden_index in range(n_Kunden):
                if kunden_index == 0:
                    load = loader(full=n_Kunden, message="predict")
                load.print_progress(kunden_index)
            kunden_vektor = KPM[kunden_index]
            kunden_buy_list = np.argwhere(kunden_vektor == 1)[:, 0]
            for prod_index in range(n_Produkte):
                P_y = prod_occ[prod_index]

                if method == "Naive":
                    P_x = 1
                    for index in kunden_buy_list:
                        P_x *= prod_occ[index]
                elif method == "Approx":
                    P_x = np.sum(KPM[:, kunden_buy_list], axis=None) / (len(kunden_buy_list)*n_Kunden)
                elif method == "Squared":
                    P_x = sum((np.sum(KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) ** 2) / n_Kunden
                elif method == "Empirical":
                    P_x = len(np.argwhere((np.sum(KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) == 1))/n_Kunden

                item_buy_list = np.argwhere(KPM[:,prod_index] == 1)[:, 0]
                reduced_KPM = KPM[item_buy_list]
                n_reduced_Kunden = len(reduced_KPM)
                reduced_prod_occ = self.get_prod_occ(reduced_KPM,n_reduced_Kunden)
                if method == "Naive":
                    P_x_if_y = 1
                    for index in kunden_buy_list:
                        P_x_if_y *= reduced_prod_occ[index]
                elif method == "Approx":
                    P_x_if_y = np.sum(reduced_KPM[:, kunden_buy_list], axis=None) / (len(kunden_buy_list)*n_reduced_Kunden)
                elif method == "Squared":
                    P_x_if_y = sum((np.sum(reduced_KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) ** 2) / n_reduced_Kunden
                elif method == "Empirical":
                    P_x_if_y = len(np.argwhere((np.sum(reduced_KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) == 1))/n_reduced_Kunden
                # if type(P_x_if_y * P_y / P_x) != float:
                #     print(type(P_x_if_y * P_y / P_x))
                #     print(P_x_if_y,P_y,P_x)
                predictions[kunden_index,prod_index]=P_x_if_y*P_y/P_x if P_x != 0 else 0
        return predictions

    def fit(self,train_KPM):
        self.train_KPM = train_KPM
        self.n_Kunden, self.n_Produkte = np.shape(train_KPM)
        self.prod_occ = self.get_prod_occ(train_KPM, self.n_Kunden)

    def predict(self,test_KPM,method):
        predictions = np.zeros_like(test_KPM)
        n_test_Kunden,n_Produkte = test_KPM.shape
        for kunden_index in range(n_test_Kunden):
            if kunden_index == 0:
                load = loader(full=n_test_Kunden, message="predict")
            load.print_progress(kunden_index)
          #load.print_progress(kunden_index, n_test_Kunden, "predict")

            kunden_vektor = test_KPM[kunden_index]
            kunden_buy_list = np.argwhere(kunden_vektor == 1)[:, 0]
            for prod_index in range(n_Produkte):
                P_y = self.prod_occ[prod_index]
                if method == "Naive":
                    P_x = 1
                    for index in kunden_buy_list:
                        P_x *= self.prod_occ[index]
                elif method == "Approx":
                    P_x = np.sum(self.train_KPM[:, kunden_buy_list], axis=None) / (len(kunden_buy_list)*self.n_Kunden)
                elif method == "Squared":
                    P_x = sum((np.sum(self.train_KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) ** 2) / self.n_Kunden
                elif method == "Empirical":
                    P_x = len(np.argwhere((np.sum(self.train_KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) == 1))/self.n_Kunden

                item_buy_list = np.argwhere(self.train_KPM[:,prod_index] == 1)[:, 0]
                reduced_KPM = self.train_KPM[item_buy_list]
                n_reduced_Kunden = len(reduced_KPM)
                reduced_prod_occ = self.get_prod_occ(reduced_KPM,n_reduced_Kunden)
                if n_reduced_Kunden == 0:
                    predictions[kunden_index, prod_index] = 0
                else:
                    if method == "Naive":
                        P_x_if_y = 1
                        for index in kunden_buy_list:
                            P_x_if_y *= reduced_prod_occ[index]
                    elif method == "Approx":
                        P_x_if_y = np.sum(reduced_KPM[:, kunden_buy_list], axis=None) / (len(kunden_buy_list)*n_reduced_Kunden)
                    elif method == "Squared":
                        P_x_if_y = sum((np.sum(reduced_KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) ** 2) / n_reduced_Kunden
                    elif method == "Empirical":
                        P_x_if_y = len(np.argwhere((np.sum(reduced_KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list)) == 1))/n_reduced_Kunden
                    predictions[kunden_index,prod_index]=P_x_if_y*P_y/P_x if P_x != 0 else 0

        return predictions

def do(dataset,split,fit_set,pred_set,method):
    if split == "orders":
        train_KPM = np.sign(np.load(dataset + "/npy_files/" + fit_set + "_KPM.npy"))
        test_KPM = np.sign(np.load(dataset + "/npy_files/" + pred_set + "_KPM.npy"))
    elif split == "clients":
        train_indexes = np.load(dataset + "/npy_files/" + fit_set + "_index.npy")
        test_indexes = np.load(dataset + "/npy_files/" + pred_set + "_index.npy")
        full_KPM = np.sign(np.load(dataset + "/npy_files/full_KPM.npy"))
        train_KPM = full_KPM[train_indexes]
        test_KPM = full_KPM[test_indexes]

    model = own_model()
    model.fit(train_KPM)
    prediction = model.predict(test_KPM, method)

    np.save(dataset + "/npy_files/" + "own_model_"+method+"_prediction",prediction)
    threshold = 0.5
    y_soll = test_KPM.flatten()
    y_prop = prediction.flatten()
    for threshold in np.linspace(0.1,0.6,20):
        y_pred = y_prop > threshold
        print(threshold)
        print("MSE", metrics.mean_squared_error(y_soll, y_prop))
        print("neg_log_loss", metrics.log_loss(y_soll, y_prop))
        print("Accuracy", metrics.accuracy_score(y_soll, y_pred))
        print("Precision", metrics.precision_score(y_soll, y_pred))
        print("Recall", metrics.recall_score(y_soll, y_pred))
        print("F1", metrics.f1_score(y_soll, y_pred))

        print("Confusion Matrix (tn,fp,fn,tp)")
        print(metrics.confusion_matrix(y_soll, y_pred))
        print("-"*100)


    # print(test_KPM.shape)
    # print(y_prop.shape)
    # fpr, tpr, thresholds = metrics.roc_curve(y_soll, y_prop)
    # plt.plot(fpr, tpr)
    # print(fpr)
    # print(tpr)
    # print(thresholds)
    # index = 0
    # min_dist = 100
    # for i in range(len(fpr)):
    #     dist = np.sqrt((tpr[i] - 1) ** 2 + (fpr[i]) ** 2)
    #     print(dist)
    #     if dist < min_dist:
    #         min_dist = dist
    #         index = i
    #
    # #   plt.scatter(fpr[index], tpr[index])
    # # plt.savefig(dataset+"/plots/AUC_PLOT.png")
    # print(index, "opt. threshold", str(thresholds[index]), "mit:", "fpr", fpr[index], "tpr", tpr[index],
    #       "- current Threshold", threshold)
    # print("-" * 100)

if __name__ == "__main__":
    dataset = sys.argv[1]
    split = sys.argv[2]
    fit_set = sys.argv[3]
    pred_set = sys.argv[4]
    method = sys.argv[5]
    do(dataset, split, fit_set, pred_set, method)


