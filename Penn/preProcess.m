root_folder = "S:\MS A&R\4th Sem\Thesis\Penn_Action\labels\";
savePath =  "S:\MS A&R\4th Sem\Thesis\Penn_Action\train\";
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
%iter = randi([3 2328],2328,1);
for j = 1840:2328
    i = iter(j);
    labels = [];
    filePattern = fullfile(root_folder + Files_root(i).name + "\");
    load(filePattern);
    % 4:length(Files)-7     length(Files)-7:length(Files)-4
    % 4:ceil(0.7*(length(Files)))      (ceil(0.7*(length(Files)))+1):floor(0.85*(length(Files)))
    %(floor(0.85*(length(Files)))+1):length(Files)
    temp = [x y];
    data = [data; temp];
    if(action=="baseball_pitch")
        labels(1:size(x,1),1) = 0;
    elseif(action=="baseball_swing")
        labels(1:size(x,1),1) = 1;
    elseif(action=="bench_press")
        labels(1:size(x,1),1) = 2;
    elseif(action=="bowl")
        labels(1:size(x,1),1) = 3;
    elseif(action=="clean_and_jerk")
        labels(1:size(x,1),1) = 4;
    elseif(action=="golf_swing")
        labels(1:size(x,1),1) = 5;
    elseif(action=="jumping_jacks")
        labels(1:size(x,1),1) = 6;
    elseif(action=="jump_rope")
        labels(1:size(x,1),1) = 7;
    elseif(action=="pullup")
        labels(1:size(x,1),1) = 8;
    elseif(action=="pushup")
        labels(1:size(x,1),1) = 9;
    elseif(action=="situp")
        labels(1:size(x,1),1) = 10;
    elseif(action=="squat")
        labels(1:size(x,1),1) = 11;
    elseif(action=="strum_guitar")
        labels(1:size(x,1),1) = 12;
    elseif(action=="tennis_forehand")
        labels(1:size(x,1),1) = 13;
    elseif(action=="tennis_serve")
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
writematrix(data,[savePath + 'test_data.csv']);