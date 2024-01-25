from sklearn.model_selection import KFold 
from sklearn import metrics

def run_kfold(machine,data,target,n,continuous):
	kfold_object = KFold(n_splits=n)
	kfold_object.get_n_splits(data)

	all_return_values = []
	i=0
	for train_index, test_index in kfold_object.split(data):
		i = i+1
		# print("Round:",str(i))
		# print("Training index:")
		# print(train_index)
		# print("Testing index:")
		# print(test_index)

		data_train = data[train_index]
		target_train = target[train_index]
		data_test = data[test_index]
		target_test = target[test_index]

		machine.fit(data_train, target_train)
		prediction = machine.predict(data_test)

		if (continuous==True):
			r2 = metrics.r2_score(target_test, prediction)
			# print("R2 score:", r2)
			# print("\n\n")
			all_return_values.append(r2)
		else:
			accuracy_score = metrics.accuracy_score(target_test, prediction)
			# print("Accuracy score:", accuracy_score)
			all_return_values.append(accuracy_score)

			confusion_matrix = metrics.confusion_matrix(target_test, prediction)
			# print("Confusion matrix:")
			# print(accuracy_score)
			# print("\n\n")
		return all_return_values
