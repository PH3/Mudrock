import pandas
import math
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import OrderedDict
from sklearn.preprocessing import RobustScaler
import os

#read the data
filelist_path = 'D:\pytorchproj\sampledata\sampleforml'
filename_list = os.listdir(filelist_path)
filepath = []
polluted_features = []
polluted_targets = [0 if _ < 640 else 1 for _ in range(800)]
print(polluted_targets.count(1))
list_per = []
per_means = OrderedDict()
for f in filename_list:
    filepath.append(os.path.join(filelist_path, f))
print(filepath)
for file in filepath:
    df = pandas.read_excel(file)
    for c in df.columns:
        polluted_features.append(df[c].values.tolist())

#calculate the performance params
def _get_model_performance_params(preds, labels):
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(labels)):
        if preds[i] == 0 and labels[i] == 0:
            tn += 1
        if preds[i] == 1 and labels[i] == 1:
            tp += 1
        if preds[i] == 0 and labels[i] == 1:
            fn += 1
        if preds[i] == 1 and labels[i] == 0:
            fp += 1
    print(tn, tp, fn, fp)
    hsi_sensitivity = tp / (tp + fn)
    hsi_specificity = tn / (tn + fp)
    hsi_precision = tp / (tp + fp)
    hsi_accuracy = (tp+tn)/(tn+tp+fn+fp)
    hsi_f1_score = 2.0 * hsi_precision * hsi_sensitivity / (hsi_precision + hsi_sensitivity)
    hsi_tmp_mcc = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    hsi_mcc = (tp * tn - fp * fn) / (math.sqrt(hsi_tmp_mcc))
    hsi_polluted_error = fp / (tn+fp)
    list_tmp_per = [{'hsi_sensitivity': hsi_sensitivity},
                    {'hsi_specificity': hsi_specificity},
                    {'hsi_precision': hsi_precision},
                    {'hsi_accuracy': hsi_accuracy},
                    {'hsi_f1_score': hsi_f1_score},
                    {'hsi_mcc': hsi_mcc},
                    {'hsi_polluted_error': hsi_polluted_error}
                    ]
    return list_tmp_per


#save the results to csv
def save_performance_means(list_per_all, model):
    model = model+'-result'
    result_performance = []
    for cn in range(len(list_per_all)):
        for i in range(len(list_per_all[cn])):
            tmp_keys = list(list_per_all[cn][i].keys())[0]
            if cn == 0:
                per_means[tmp_keys] = 0
                per_means[tmp_keys] += list_per_all[cn][i][tmp_keys]
            else:
                per_means[tmp_keys] += list_per_all[cn][i][tmp_keys]
        print(per_means)
    for n in per_means.keys():
        per_means[n] = per_means[n]/len(list_per_all)
    result_performance.append(per_means)
    pandas.DataFrame.from_dict(result_performance, orient='columns').to_csv(f'{model}.csv', index_label=model, mode='a+')


#calculate avg of random dataset
def calculate_means(model, dataset, targets, selector):
    list_performance_all = []
    for s in range(30):
        dataset_feature_train, dataset_feature_test, dataset_target_train, dataset_target_test = train_test_split(
            dataset, targets, test_size=0.2, stratify=targets)
        model.fit(dataset_feature_train, dataset_target_train)
        predict_results = model.predict(dataset_feature_test)
        print('accuracy_' + selector, accuracy_score(predict_results, dataset_target_test), '  Turn', s)
        tmp_list = _get_model_performance_params(predict_results, dataset_target_test)
        list_performance_all.append(tmp_list)
    print(len(list_performance_all))
    save_performance_means(list_performance_all, selector)


#RobustScaler
mm = RobustScaler()
polluted_features = mm.fit_transform(polluted_features)

clf = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(150, ), alpha=0.01, random_state=0, max_iter=4000)
calculate_means(clf, polluted_features, polluted_targets, 'mlp')
ada = AdaBoostClassifier(algorithm="SAMME.R")
calculate_means(ada, polluted_features, polluted_targets, 'adaboost')
svm1 = SVC(kernel='poly', C=2.0, random_state=None, class_weight='balanced')
calculate_means(svm1, polluted_features, polluted_targets, 'svm')



