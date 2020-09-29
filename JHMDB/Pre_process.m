root_folder = "S:\MS A&R\4th Sem\Thesis\J-HMDB\joint_positions\joint_positions\";
savePath =  "S:\MS A&R\4th Sem\Thesis\J-HMDB\joint_positions\train\";
%if ~isdir(myFolder)
 % errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  %uiwait(warndlg(errorMessage));
  %return;
%end

filePattern_root = fullfile(root_folder, '*');
Files_root = dir(filePattern_root);

data = [];
for i = 4:length(Files_root)
    
    filePattern = fullfile(root_folder + Files_root(i).name + "\");
    Files = dir(filePattern);
  
    % 4:ceil(0.7*(length(Files)))      (ceil(0.7*(length(Files)))+1):floor(0.85*(length(Files)))
    for k = (floor(0.85*(length(Files)))+1):length(Files)
      if(Files(k).name == ".DS_Store")
        continue;
      end
      labels = [];
      matFiles = filePattern + Files(k).name + "\";
      mfile = fullfile(matFiles, '*.mat');
      F = dir(mfile);
      baseFileName = F.name;
      fullFileName = fullfile(matFiles, baseFileName);
      fprintf(1, 'Now reading %s\n', fullFileName);
      matData = load(fullFileName);
      pos = matData.pos_img;
      labels(1:size(pos,3),1) = i - 5;
      t =reshape(pos,[size(pos,1)*size(pos,2),size(pos,3)]);
      t = transpose(t);
      t = [t, labels];
      data = [data;t];
      
    end
    
end
time = transpose(linspace(0, 0.04*size(data,1),size(data,1)));
data = [time,data];     
empt = zeros([1,size(data,2)]);
data = [empt;data];
writematrix(data,[savePath + 'test_data.csv']);