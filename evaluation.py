from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import json
from tools.load import loader
import sys

def eval(dataset,prediction_filename,split,set,threshold=0.5,top_n_test=True,top_n=20):

	index = 0
	min_dist = 1

	if split == "clients":
		indexes = np.load(dataset+"/npy_files/"+set+"_index.npy")
		KPM = np.load(dataset+"/npy_files/full_KPM.npy")[indexes] == 1
	elif split == "orders":
		KPM = np.load(dataset+"/npy_files/"+ set + "_KPM.npy") == 1

	n_Kunden, n_Produkte = KPM.shape

	print("Kunden",n_Kunden)
	print("Produkte",n_Produkte)
	print("Interaktionen",np.sum(np.sum(KPM)))

	if threshold < 0:
		threshold = 1/n_Produkte

	predictions = np.load(dataset+"/npy_files/"+prediction_filename)
	calssifications = predictions.flatten() > threshold

	if top_n_test:
		n_orders = np.sum(KPM, axis=None)
		n_hits = 0
		for client_index in range(n_Kunden):
			if client_index == 0:
				load = loader(len(predictions), "evaluation")
			load.print_progress(client_index)
			bought_items = np.argwhere(KPM[client_index] == 1)[:, 0]
			for item_index in bought_items:
				if item_index in np.array(
						sorted(zip(predictions[client_index], np.arange(len(predictions[client_index]))), reverse=True))[:,
								 1][:top_n]:
					n_hits += 1
		score = n_hits / n_orders
		print(str(score * 100) + "%\t(", n_hits, "von", n_orders, ") \tder getätigten käufte sind in der top", top_n,
			  "der Produktempfehlungen")

	KPM = KPM.flatten()
	print("Kpm flattened")
	#fpr, tpr, thresholds = metrics.roc_curve(KPM, predictions.flatten())

	print(prediction_filename+":")
	print("MSE",metrics.mean_squared_error(KPM, predictions.flatten()))
	print("neg_log_loss", metrics.log_loss(KPM, predictions.flatten()))
	print("Accuracy",metrics.accuracy_score(KPM, calssifications))
	print("Precision", metrics.precision_score(KPM, calssifications))
	print("Recall", metrics.recall_score(KPM, calssifications))
	print("F1", metrics.f1_score(KPM, calssifications))

	print("Confusion Matrix (tn,fp,fn,tp)")
	print(metrics.confusion_matrix(KPM, calssifications))

	#plt.plot(fpr,tpr,label=prediction_filename+"_auc_score:"+str(metrics.roc_auc_score(KPM, predictions)))


	# if roc_opt_mode != "None":
	# 	for i in range(len(fpr)):
	# 		if roc_opt_mode == "dist": # erhöht den recall und verringert die precsion
	# 			dist = np.sqrt((tpr[i]-1)**2+(fpr[i])**2)
	# 			if dist < min_dist:
	# 				min_dist = dist
	# 				index = i
	# 		elif roc_opt_mode == "precision": #funktioniert nicht da fpr/tpr rate anstatt value berechnung des values dauert sehr lange da confusion matrix
	# 			dist = tpr[i]/(fpr[i]+tpr[i])
	# 			if dist > min_dist:
	# 				min_dist = dist
	# 				index = i
	#
	# plt.scatter(fpr[index],tpr[index])
	# plt.savefig(dataset+"/plots/AUC_PLOT.png")
	# print("opt. threshold",str(thresholds[index]),"mit:","fpr",fpr[index],"tpr",tpr[index],"- current Threshold", threshold)
	# print("-"*100)

def do(dataset):
	print("load config")
	with open("{dataset}/json_files/config.json".format(dataset=dataset.strip()), 'r') as fp:
		config = json.load(fp)
	# title = config["dataset"] + "_predictions_" + "fit" + config["fit_set"] + \
	# 		"_pred" + config["pred_set"] + "_" + config["NaiveBayes"]["model_type"] + \
	# 		"_approach" + str(config["approach"]) + "_split" + config["split"] + \
	# 		"_count" + str(config["count"]) + \
	# 		"_info" + str(config["use_user_info"]) + config["info_string"]
	title = config["model_name"] + "_predictions"
	predictions_filename = title+".npy"
	split = config["split"]
	set = config["pred_set"]
	if config["approach"] == "binary":
		threshold = 0.5
	else:
		threshold = 1/config["n_Produkte"]

	eval(dataset, predictions_filename, split, set, threshold)

if __name__=="__main__":

	dataset = sys.argv[1].strip()
	do(dataset)
