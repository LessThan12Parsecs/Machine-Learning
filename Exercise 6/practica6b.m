%Practica 6b Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Deteccion de Spam con SVM
%Realizado con los correos en spam, easy_ham, hard_ham.zip
%inciamos una matriz de tama?o numeroDeEmails*componentesVocab
m = 3301;
n = 1899;
emails = zeros(m,n);
y = zeros(m,1);
%leemos y cargamos los correos de spam
directorio = 'spam';
for i = 1:500
    file_name = sprintf('%s/%04d.txt', directorio, i);
    file_contents = readFile(file_name);
    email = processEmail(file_contents);
    emails(i,:) = email';
    y(i,:) = 1;
end

%leemos y cargamos  los correos en easy ham
directorio = 'easy_ham';

for i = 1:2551
    file_name = sprintf('%s/%04d.txt', directorio, i);
    file_contents = readFile(file_name);
    email = processEmail(file_contents);
    emails(i+500,:) = email';
    y(i+500,:) = 0;
end

%leemos y cargamos  los correos en hard ham
directorio = 'hard_ham';
for i = 1:250
      file_name = sprintf('%s/%04d.txt', directorio, i);
    file_contents = readFile(file_name);
    email = processEmail(file_contents);
    emails(i+3051,:) = email';
    y(i+3051,:) = 0;
end


emails_train = vertcat(emails(51:699,:),emails(1056:3051,:),emails(3078:3301,:));
y_train = vertcat(y(51:699,:),y(1056:3051,:),y(3078:3301,:));
emails_valid = vertcat(emails(1:50,:),emails(700:1055,:),emails(3052:3077,:));
y_valid = vertcat(y(1:50,:),y(700:1055,:),y(3052:3077,:));


valores = [0.03,0.1,0.3,10];
percent = zeros(4,1);

for iC = 1:length(valores)
    model = svmTrain(emails_train,y_train,iC, @linearKernel,1e-3,20);
    prediction = svmPredict(model,emails_valid);
    success = (prediction == y_valid);
    percent(iC) = sum(success)/length(success);
end


