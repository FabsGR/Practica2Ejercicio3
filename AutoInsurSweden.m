% Cargar tus datos desde un archivo CSV
data = readtable('AutoInsurSwedenDataset.csv');  % Ajusta el nombre del archivo

% Obtener las características (X) y las etiquetas de clase (Y)
X = data.X;
Y = data.Y;

% Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento)
n = length(X);  % Número total de ejemplos
n_train = round(0.8 * n);  % 80% para entrenamiento
n_test = n - n_train;  % Resto para prueba

X_train = X(1:n_train);
Y_train = Y(1:n_train);

X_test = X(n_train + 1:end);
Y_test = Y(n_train + 1:end);

% Regresión Logística
mdl_logistica = fitmnr(X_train, Y_train);
Y_test_pred_logistica = predict(mdl_logistica, X_test);
conf_mat_logistica = confusionmat(Y_test, Y_test_pred_logistica);
accuracy_logistica = sum(diag(conf_mat_logistica)) / sum(conf_mat_logistica(:));
precision_logistica = conf_mat_logistica(2, 2) / sum(conf_mat_logistica(:, 2));
sensitivity_logistica = conf_mat_logistica(2, 2) / sum(conf_mat_logistica(2, :));
specificity_logistica = conf_mat_logistica(1, 1) / sum(conf_mat_logistica(1, :));
f1_score_logistica = 2 * (precision_logistica * sensitivity_logistica) / (precision_logistica + sensitivity_logistica);

% K-Nearest Neighbors (K-NN)
k = 5;  % Ajusta el valor de k según tus necesidades
mdl_knn = fitcknn(X_train, Y_train, 'NumNeighbors', k);
Y_test_pred_knn = predict(mdl_knn, X_test);
conf_mat_knn = confusionmat(Y_test, Y_test_pred_knn);
accuracy_knn = sum(diag(conf_mat_knn)) / sum(conf_mat_knn(:));
precision_knn = conf_mat_knn(2, 2) / sum(conf_mat_knn(:, 2));
sensitivity_knn = conf_mat_knn(2, 2) / sum(conf_mat_knn(2, :));
specificity_knn = conf_mat_knn(1, 1) / sum(conf_mat_knn(1, :));
f1_score_knn = 2 * (precision_knn * sensitivity_knn) / (precision_knn + sensitivity_knn);


% Support Vector Machines (SVM)
mdl_svm = fitcecoc(X_train, Y_train);
Y_test_pred_svm = predict(mdl_svm, X_test);
conf_mat_svm = confusionmat(Y_test, Y_test_pred_svm);
accuracy_svm = sum(diag(conf_mat_svm)) / sum(conf_mat_svm(:));
precision_svm = conf_mat_svm(2, 2) / sum(conf_mat_svm(:, 2));
sensitivity_svm = conf_mat_svm(2, 2) / sum(conf_mat_svm(2, :));
specificity_svm = conf_mat_svm(1, 1) / sum(conf_mat_svm(1, :));
f1_score_svm = 2 * (precision_svm * sensitivity_svm) / (precision_svm + sensitivity_svm);

% Naive Bayes
mdl_naive_bayes = fitcecoc(X_train, Y_train);
Y_test_pred_naive_bayes = predict(mdl_naive_bayes, X_test);
conf_mat_naive_bayes = confusionmat(Y_test, Y_test_pred_naive_bayes);
accuracy_naive_bayes = sum(diag(conf_mat_naive_bayes)) / sum(conf_mat_naive_bayes(:));
precision_naive_bayes = conf_mat_naive_bayes(2, 2) / sum(conf_mat_naive_bayes(:, 2));
sensitivity_naive_bayes = conf_mat_naive_bayes(2, 2) / sum(conf_mat_naive_bayes(2, :));
specificity_naive_bayes = conf_mat_naive_bayes(1, 1) / sum(conf_mat_naive_bayes(1, :));
f1_score_naive_bayes = 2 * (precision_naive_bayes * sensitivity_naive_bayes) / (precision_naive_bayes + sensitivity_naive_bayes);

% Mostrar resultados de precisión y otras métricas
disp(['Regresión Logística:']);
disp(['Accuracy: ' num2str(accuracy_logistica * 100) '%']);
disp(['Precision: ' num2str(precision_logistica * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_logistica * 100) '%']);
disp(['Specificity: ' num2str(specificity_logistica * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_logistica * 100) '%']);

disp(['K-NN:']);
disp(['Accuracy: ' num2str(accuracy_knn * 100) '%']);
disp(['Precision: ' num2str(precision_knn * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_knn * 100) '%']);
disp(['Specificity: ' num2str(specificity_knn * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_knn * 100) '%']);

disp(['SVM:']);
disp(['Accuracy: ' num2str(accuracy_svm * 100) '%']);
disp(['Precision: ' num2str(precision_svm * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_svm * 100) '%']);
disp(['Specificity: ' num2str(specificity_svm * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_svm * 100) '%']);

disp(['Naive Bayes:']);
disp(['Accuracy: ' num2str(accuracy_naive_bayes * 100) '%']);
disp(['Precision: ' num2str(precision_naive_bayes * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_naive_bayes * 100) '%']);
disp(['Specificity: ' num2str(specificity_naive_bayes * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_naive_bayes * 100) '%']);
