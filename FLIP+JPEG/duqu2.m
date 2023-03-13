

function duqu2(imgDataPath,save_path,Q)

    imgDataDir = dir(imgDataPath); % 遍历所有文件
%     B = 0;
%     total = 0;
    for i = 3:length(imgDataDir)
        imgDir = dir(strcat(strcat(imgDataPath,"/"),imgDataDir(i).name));
        for j = 3:length(imgDir) % 遍历所有图片
            image_path = strcat('',strcat(strcat(strcat(strcat(imgDataPath,"/"),imgDataDir(i).name),strcat("/",imgDir(j).name)),''));

                disp(image_path)
                image = imread(image_path);
                R_path = strcat(strcat(strcat(strcat(save_path,"/"),imgDir(j).name(1:9)),"/"),imgDir(j).name);
%                 mask1 = bianyuantiqu(image_path);
% % %                 
%                 image = mask(image,mask1,Q);
%                 image = jpeg(image,Q);
%                 image = shuangbian(image);
                image = fanzhuan(image);
%                   image = zengqiang(image_path);
                image = jpeg(image,Q);

                
%                 image = rgb_eq(image);
%                 image = pinghua(image);
       
                if ~exist(strcat(strcat(strcat(save_path,"/"),imgDir(j).name(1:9)),"/"),'dir')
                     mkdir(strcat(strcat(strcat(save_path,"/"),imgDir(j).name(1:9))),"/");
                     disp("creat success")
                end
                imwrite(image,R_path);

               

        end  
    end