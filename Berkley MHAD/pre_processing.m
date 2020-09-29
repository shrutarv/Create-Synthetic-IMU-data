
root_folder = "S:\MS A&R\4th Sem\Thesis\Berkley MHAD\SkeletalData-20200922T160342Z-001\SkeletalData_validation\";
savePath =  "S:\MS A&R\4th Sem\Thesis\Berkley MHAD\SkeletalData-20200922T160342Z-001\train\";
%if ~isdir(myFolder)
 % errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  %uiwait(warndlg(errorMessage));
  %return;
%end

filePattern_root = fullfile(root_folder, '*');
Files_root = dir(filePattern_root);
final_data = [];
for j = 4:length(Files_root)
   
    labels = [];
    fprintf(1, 'Now reading %s\n', Files_root(j).name);
    filePattern = fullfile(root_folder + Files_root(j).name);
    %Files = dir(filePattern);
    name = Files_root(j).name;
    
    [skeleton,time] = loadbvh(name);
    if(contains(name,"a01"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 0;
    elseif(contains(name,"a02"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 1;
    elseif(contains(name,"a03"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 2;
    elseif(contains(name,"a04"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 3;
    elseif(contains(name,"a05"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 4;
    elseif(contains(name,"a06"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 5;
    elseif(contains(name,"a07"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 6;
    elseif(contains(name,"a08"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 7;
    elseif(contains(name,"a09"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 8;
    elseif(contains(name,"a10"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 9;
    elseif(contains(name,"a11"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 10;
    elseif(contains(name,"t-pose"))
        temp = skeleton.Dxyz;
        labels(1:size(temp,2),1) = 11;
    end
    data = [];
    for i = 1:34
        D_xyz = skeleton(i).Dxyz;
        D_xyz = transpose(D_xyz);
        data = [data D_xyz];
    end
    data = [data labels];
    final_data = [final_data;data];
end
time = transpose(linspace(0, (1/480)*size(final_data,1),size(final_data,1)));
data_save = [time,final_data];     
empt = zeros([1,size(data_save,2)]);
data_save = [empt;data_save];
writematrix(data_save,[savePath + 'validation_data.csv']);