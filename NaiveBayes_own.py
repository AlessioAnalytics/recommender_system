import pandas as pd
import numpy as np
from tools import npy_to_csv
from tools import load
from retrieve_user_information import get_info


class NaiveBayes:
    def __init__(self, path_to_data="csv_files/superstore.csv", user_info_koeff=0, squared=False, set_already_bough_prop = False, make_model=True):
        self.path_to_data = path_to_data
        if make_model:
            self.data = self.get_data(path_to_data)
            self.get_KPM()
            # self.KPM = np.array([[0,0,0,1,1,0,0,1],[1,1,1,0,1,1,1,0],[0,1,0,1,1,0,0,0],[0,0,0,1,0,1,0,1],[1,1,1,1,0,0,1,0]])
            # self.n_Produkte=8
            # self.n_Kunden=5
            self.user_info_koeff = user_info_koeff
            self.squared = squared
            self. set_already_bough_prop= set_already_bough_prop
            self.get_occ_matricies(True)
            self.calc_client_prop(True, user_info_koeff=user_info_koeff, squared=squared, set_already_bough_prop=set_already_bough_prop)
            self.get_recommendations(5)
            self.save_vals()

    def get_data(self, path_to_data):
        data = pd.read_csv(path_to_data).values
        data = np.delete(data, 2, axis=1)
        return data

    def get_KPM(self):
        self.n_Produkte = int(np.max(self.data[:, 1])) + 1
        self.n_Kunden = int(np.max(self.data[:, 0])) + 1

        Kunden_Produkte_Matrix = np.zeros((self.n_Kunden, self.n_Produkte))
        for i in self.data:
            Kunden_Produkte_Matrix[int(i[0]), int(i[1])] = 1

        self.KPM = Kunden_Produkte_Matrix
        return Kunden_Produkte_Matrix

    def get_occ_matricies(self, show_progress=False):
        if_occurence = np.zeros((self.n_Produkte, self.n_Produkte))
        occurence = np.zeros(self.n_Produkte)
        print("get_occ_matricies:","."*100)
        for row in range(self.n_Produkte):
            occurence[row] = sum(self.KPM[:, row]) / self.n_Kunden
            if show_progress:
                load.print_progress(row / self.n_Produkte,"get_occ_matricies")
            for col in range(row + 1):
                p_row = occurence[row]
                p_col = occurence[col]
                p_row_and_col = self.KPM[:, row].dot(self.KPM[:, col]) / self.n_Kunden
                # if_occurence[row,col]=P(row|col)
                if_occurence[row, col] = p_row_and_col / p_col if p_col != 0 else 0
                if_occurence[col, row] = p_row_and_col / p_row if p_row != 0 else 0

        self.if_occ = if_occurence
        self.occ = occurence
        return if_occurence, occurence

    def calc_client_prop(self, show_progress=False,user_info_koeff=0,squared = True,set_already_bough_prop=False):
        client_prop = np.zeros((self.n_Kunden, self.n_Produkte))
        print("Calculate Propabilities:","."*100)

        if user_info_koeff != 0:
            user_info = get_info()
            user_info_matrix=(user_info.dot(user_info.T) / 5)# 5 ist die anzahl der one hot encodeten features

        for kunden_index in range(self.n_Kunden):

            if show_progress:
                load.print_progress(kunden_index / self.n_Kunden,"Calculate Propabilities")

            kunden_vektor = self.KPM[kunden_index]

            kunden_buy_list = np.argwhere(kunden_vektor == 1)[:, 0]

            for produkt_index in range(self.n_Produkte):
                if produkt_index in kunden_buy_list and set_already_bough_prop:
                    client_prop[kunden_index, produkt_index] = set_already_bough_prop
                else:
                    P_x = sum(np.sum(self.KPM[:,kunden_buy_list],axis=1)/len(kunden_buy_list)**2)/self.n_Kunden

                    P_y = self.occ[produkt_index]
                                    #P_x_if_y = sum(
                                     #   np.sum(self.KPM[np.argwhere(self.KPM[:, produkt_index] == 1)[:, 0]][:, kunden_buy_list],
                                      #         axis=1) / len(kunden_buy_list) ** 2) / self.n_Kunden
                    if squared:
                        P_x_if_y = np.sum(self.KPM[np.argwhere(self.KPM[:,produkt_index]==1)[:,0]][:,kunden_buy_list],axis=1)/len(kunden_buy_list)**2
                    else:
                        P_x_if_y = np.sum(self.KPM[np.argwhere(self.KPM[:, produkt_index] == 1)[:, 0]][:, kunden_buy_list],axis=1)/len(kunden_buy_list)

                    if user_info_koeff != 0:
                        P_x_if_y=(1-user_info_koeff)*P_x_if_y + user_info_koeff* P_x_if_y*user_info_matrix[kunden_index,np.argwhere(self.KPM[:,produkt_index]==1)[:,0]]

                    P_x_if_y= sum(P_x_if_y)/self.n_Kunden
                    client_prop[kunden_index, produkt_index] = P_x_if_y*P_y /P_x

        self.client_prop = client_prop

    def get_recommendations(self,n_recs=20):
        recs = np.zeros((self.n_Kunden,n_recs))
        for kunden_index in range(self.n_Kunden):
            recs[kunden_index]=np.array(sorted(zip(self.client_prop[kunden_index],np.arange(self.n_Produkte)),reverse=True)[:n_recs])[:,1]
        self.recs=recs

    def predict(self,kunden_vektor,kunden_user_info_vektor=0):
        client_prop = np.zeros(self.n_Produkte)
        kunden_buy_list = np.argwhere(kunden_vektor == 1)[:, 0]
        if kunden_user_info_vektor:
            user_info = get_info().dot(kunden_user_info_vektor)
        for produkt_index in range(self.n_Produkte):
            if produkt_index in kunden_buy_list and self.set_already_bough_prop:
                client_prop[produkt_index] = self.set_already_bough_prop
            else:
                P_x = sum(np.sum(self.KPM[:, kunden_buy_list], axis=1) / len(kunden_buy_list) ** 2) / self.n_Kunden

                P_y = self.occ[produkt_index]
                # P_x_if_y = sum(
                #   np.sum(self.KPM[np.argwhere(self.KPM[:, produkt_index] == 1)[:, 0]][:, kunden_buy_list],
                #         axis=1) / len(kunden_buy_list) ** 2) / self.n_Kunden
                if self.squared:
                    P_x_if_y = np.sum(self.KPM[np.argwhere(self.KPM[:, produkt_index] == 1)[:, 0]][:, kunden_buy_list],
                                      axis=1) / len(kunden_buy_list) ** 2
                else:
                    P_x_if_y = np.sum(self.KPM[np.argwhere(self.KPM[:, produkt_index] == 1)[:, 0]][:, kunden_buy_list],
                                      axis=1) / len(kunden_buy_list)

                if self.user_info_koeff != 0:
                    P_x_if_y = (1 - self.user_info_koeff) * P_x_if_y + self.user_info_koeff * P_x_if_y * user_info[np.argwhere(self.KPM[:, produkt_index] == 1)[:, 0]]

                P_x_if_y = sum(P_x_if_y) / self.n_Kunden
                client_prop[produkt_index] = P_x_if_y * P_y / P_x
        return client_prop

    def save_vals(self):
        np.save("npy_files/KPM.npy", self.KPM)
        np.save("npy_files/occ.npy", self.occ)
        np.save("npy_files/if_occ.npy", self.if_occ)
        print(self.client_prop)

        np.save("npy_files/client_prop_sq" + str(self.squared) + "_koeff" + str(self.user_info_koeff) + "_penal" + str(
            self.set_already_bough_prop) + ".npy",
                self.client_prop)
        npy_to_csv.npy_to_csv(
            "npy_files/client_prop_sq" + str(self.squared) + "_koeff" + str(self.user_info_koeff) + "_penal" + str(
                self.set_already_bough_prop) + ".npy",
            "csv_files/client_prop_sq" + str(self.squared) + "_koeff" + str(self.user_info_koeff) + "_penal" + str(
                self.set_already_bough_prop) + ".csv",
            "Item",
            "Client")
        np.save("npy_files/recs_sq" + str(self.squared) + "_koeff" + str(self.user_info_koeff) + "_penal" + str(
                self.set_already_bough_prop) + ".npy",
                self.recs)
        npy_to_csv.npy_to_csv(
            "npy_files/recs_sq" + str(self.squared) + "_koeff" + str(self.user_info_koeff) + "_penal" + str(
                self.set_already_bough_prop) + ".npy",
            "csv_files/recs_sq" + str(self.squared) + "_koeff" + str(self.user_info_koeff) + "_penal" + str(
                self.set_already_bough_prop) + ".csv",
            "Item",
            "Client")

if __name__ == "__main__":
    nb = NaiveBayes()





# nb = NaiveBayes()
# np.save("npy_files/KPM.npy",nb.KPM)
# np.save("npy_files/occ.npy",nb.occ)
# np.save("npy_files/if_occ.npy", nb.if_occ)
# print(nb.client_prop)
# np.save("npy_files/client_prop.npy", nb.client_prop)
# npy_to_csv.npy_to_csv("npy_files/client_prop.npy", "csv_files/client_prop.csv", "Item", "Client")
# np.save("npy_files/recs.npy", nb.recs)
# npy_to_csv.npy_to_csv("npy_files/recs.npy", "csv_files/recs.csv", "Item", "Client")
