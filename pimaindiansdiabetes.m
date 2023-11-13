% Cargar tus datos desde un archivo CSV o cualquier fuente de datos
data = readtable('pimaindiansdiabetesDataset.csv');  % Ajusta el nombre del archivo

% Separar las características (X) y las etiquetas de clase (y)
X = table2array(data(:, 1:8));  % Selecciona las primeras 8 columnas como características
y = data.Var9;  % Ajusta el nombre de la columna que contiene las etiquetas de clase

% Obtener el número total de ejemplos
n = size(X, 1);

% Determinar el porcentaje de datos para entrenamiento (80%)
train_percentage = 0.8;

% Generar una permutación aleatoria de los índices
rng('default');  % Establecer la semilla aleatoria para reproducibilidad
random_indices = randperm(n);

% Calcular el número de ejemplos para entrenamiento
n_train = round(train_percentage * n);
n_test = n - n_train;  % Resto para prueba

% Obtener los índices para los datos de entrenamiento y prueba
train_indices = random_indices(1:n_train);
test_indices = random_indices(n_train + 1:end);

% Dividir los datos en conjuntos de entrenamiento y prueba
X_train = X(train_indices, :);
y_train = y(train_indices);

X_test = X(test_indices, :);
y_test = y(test_indices);

% Regresión Logística Multiclase
mdl_logistica = fitmnr(X_train, y_train);
Y_test_pred_logistica = predict(mdl_logistica, X_test);
conf_mat_logistica = confusionmat(y_test, Y_test_pred_logistica);
accuracy_logistica = sum(diag(conf_mat_logistica)) / sum(conf_mat_logistica(:));
precision_logistica = conf_mat_logistica(2, 2) / sum(conf_mat_logistica(:, 2));
sensitivity_logistica = conf_mat_logistica(2, 2) / sum(conf_mat_logistica(2, :));
specificity_logistica = conf_mat_logistica(1, 1) / sum(conf_mat_logistica(1, :));
f1_score_logistica = 2 * (precision_logistica * sensitivity_logistica) / (precision_logistica + sensitivity_logistica);

disp(['Regresión Logística:']);
disp(['Accuracy: ' num2str(accuracy_logistica * 100) '%']);
disp(['Precision: ' num2str(precision_logistica * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_logistica * 100) '%']);
disp(['Specificity: ' num2str(specificity_logistica * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_logistica * 100) '%']);

% K-Nearest Neighbors (K-NN)
k = 5;  % Ajusta el valor de k según tus necesidades
mdl_knn = fitcknn(X_train, y_train, 'NumNeighbors', k);
Y_test_pred_knn = predict(mdl_knn, X_test);
conf_mat_knn = confusionmat(y_test, Y_test_pred_knn);
accuracy_knn = sum(diag(conf_mat_knn)) / sum(conf_mat_knn(:));
precision_knn = conf_mat_knn(2, 2) / sum(conf_mat_knn(:, 2));
sensitivity_knn = conf_mat_knn(2, 2) / sum(conf_mat_knn(2, :));
specificity_knn = conf_mat_knn(1, 1) / sum(conf_mat_knn(1, :));
f1_score_knn = 2 * (precision_knn * sensitivity_knn) / (precision_knn + sensitivity_knn);

disp(['K-NN:']);
disp(['Accuracy: ' num2str(accuracy_knn * 100) '%']);
disp(['Precision: ' num2str(precision_knn * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_knn * 100) '%']);
disp(['Specificity: ' num2str(specificity_knn * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_knn * 100) '%']);

% Support Vector Machines (SVM)
mdl_svm = fitcsvm(X_train, y_train);
Y_test_pred_svm = predict(mdl_svm, X_test);
conf_mat_svm = confusionmat(y_test, Y_test_pred_svm);
accuracy_svm = sum(diag(conf_mat_svm)) / sum(conf_mat_svm(:));
precision_svm = conf_mat_svm(2, 2) / sum(conf_mat_svm(:, 2));
sensitivity_svm = conf_mat_svm(2, 2) / sum(conf_mat_svm(2, :));
specificity_svm = conf_mat_svm(1, 1) / sum(conf_mat_svm(1, :));
f1_score_svm = 2 * (precision_svm * sensitivity_svm) / (precision_svm + sensitivity_svm);

disp(['SVM:']);
disp(['Accuracy: ' num2str(accuracy_svm * 100) '%']);
disp(['Precision: ' num2str(precision_svm * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_svm * 100) '%']);
disp(['Specificity: ' num2str(specificity_svm * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_svm * 100) '%']);

% Naive Bayes
mdl_naive_bayes = fitcnb(X_train, y_train);
Y_test_pred_naive_bayes = predict(mdl_naive_bayes, X_test);
conf_mat_naive_bayes = confusionmat(y_test, Y_test_pred_naive_bayes);
accuracy_naive_bayes = sum(diag(conf_mat_naive_bayes)) / sum(conf_mat_naive_bayes(:));
precision_naive_bayes = conf_mat_naive_bayes(2, 2) / sum(conf_mat_naive_bayes(:, 2));
sensitivity_naive_bayes = conf_mat_naive_bayes(2, 2) / sum(conf_mat_naive_bayes(2, :));
specificity_naive_bayes = conf_mat_naive_bayes(1, 1) / sum(conf_mat_naive_bayes(1, :));
f1_score_naive_bayes = 2 * (precision_naive_bayes * sensitivity_naive_bayes) / (precision_naive_bayes + sensitivity_naive_bayes);

disp(['Naive Bayes:']);
disp(['Accuracy: ' num2str(accuracy_naive_bayes * 100) '%']);
disp(['Precision: ' num2str(precision_naive_bayes * 100) '%']);
disp(['Sensitivity: ' num2str(sensitivity_naive_bayes * 100) '%']);
disp(['Specificity: ' num2str(specificity_naive_bayes * 100) '%']);
disp(['F1 Score: ' num2str(f1_score_naive_bayes * 100) '%']);
