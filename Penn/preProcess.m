root_folder = "S:\Datasets\Penn_Action\Penn_Action\labels\";
savePath =  "S:\Datasets\Penn_Action\Penn_Action\train\";
%if ~isdir(myFolder)
 % errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  %uiwait(warndlg(errorMessage));
  %return;
%end

filePattern_root = fullfile(root_folder, '*');
Files_root = dir(filePattern_root);
temp = [];
data = [];
label = [];
final = [];
zero=0;
one=0;
two=0;
three=0;
four=0;
five=0;
six=0;
seven=0;
eight=0;
nine=0;
ten=0;
eleven=0;
twelve=0;
thirteen = 0;
fourteen = 0;
video_size = [];
%iter = randi([3 2328],2328,1);
% train 1:1350
% validation 1350:1840
% test 1840:2326
for j = 3:2328
    i = j;
    labels = [];
    filePattern = fullfile(root_folder + Files_root(i).name + "\");
    load(filePattern);
    % 4:length(Files)-7     length(Files)-7:length(Files)-4
    % 4:ceil(0.7*(length(Files)))      (ceil(0.7*(length(Files)))+1):floor(0.85*(length(Files)))
    %(floor(0.85*(length(Files)))+1):length(Files)
    
    temp = [x y];
    data = [data; temp];
    if(action=="baseball_pitch")
        zero = zero +1;
        labels(1:size(x,1),1) = 0;
    elseif(action=="baseball_swing")
        one = one +1;
        labels(1:size(x,1),1) = 1;
    elseif(action=="bench_press")
        two = two +1;
        labels(1:size(x,1),1) = 2;
    elseif(action=="bowl")
        three = three +1;
        labels(1:size(x,1),1) = 3;
    elseif(action=="clean_and_jerk")
        four = four+1;
        labels(1:size(x,1),1) = 4;
    elseif(action=="golf_swing")
        five = five + 1;
        labels(1:size(x,1),1) = 5;
    elseif(action=="jumping_jacks")
        six = six +1;
        labels(1:size(x,1),1) = 6;
    elseif(action=="jump_rope")
        seven = seven + 1;
        labels(1:size(x,1),1) = 7;
    elseif(action=="pullup")
        eight = eight +1;
        labels(1:size(x,1),1) = 8;
    elseif(action=="pushup")
        nine =nine +1;
        labels(1:size(x,1),1) = 9;
    elseif(action=="situp")
        ten = ten+1;
        labels(1:size(x,1),1) = 10;
    elseif(action=="squat")
        eleven = eleven +1;
        labels(1:size(x,1),1) = 11;
    elseif(action=="strum_guitar")
        twelve = twelve +1;
        labels(1:size(x,1),1) = 12;
    elseif(action=="tennis_forehand")
        thirteen = thirteen + 1;
        labels(1:size(x,1),1) = 13;
    elseif(action=="tennis_serve")
        fourteen = fourteen +1;
        labels(1:size(x,1),1) = 14;
    else
        disp(i);
    end
    label = [label; labels];
    final = [data label];
end
data = final;
time = transpose(linspace(0, 0.02*size(final,1),size(final,1)));
data = [time,data];     
empt = zeros([1,size(data,2)]);
data = [empt;data];
data_imp = data;
for i=2:length(data_imp)
    data_imp(i,2:2:26) = data_imp(i,2:2:26) - data_imp(i,2);
    data_imp(i,3:2:27) = data_imp(i,3:2:27) - data_imp(i,3);
end
writematrix(data_imp,[savePath + 'train_data_tf.csv']);
data_train_val = data_imp(1:146000,:);
data_train = data_imp(1:130000,:);
data_test = data_imp(146000:163842,:);
data_val = data_imp(130001:146000,:);
data_train_75 = data_train(1:0.75*130000,:);
data_train_50 = data_train(1:0.5*130000,:);
data_train_75 = data_train(1:0.75*130000,:);

writematrix(data_train,[savePath + 'train_data.csv']);
writematrix(data_test,[savePath + 'test_data.csv']);
writematrix(data_val,[savePath + 'val_data.csv']);

writematrix(data_imp,[savePath + 'norm_data.csv']);

X = data(1,1:2:end-1);
Y = data(1,2:2:end);
n = [1:25];
figure;
scatter(X,Y,'filled');
for ii = 1: numel(n) 
    text (X(ii), Y (ii), cellstr(num2str(n(ii))));
end
