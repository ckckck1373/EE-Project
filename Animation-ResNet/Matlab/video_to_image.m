%% ?��??
index = 1;
for a = 1001:1001
video_file=strcat('D:\Onepiece\video\', num2str(a), '.mp4');

video=VideoReader(video_file);

frame_number=floor(video.Duration * video.FrameRate);

%% ����?��
  for i=1:1000:frame_number
  
  if index < 10
    image_name=strcat('D:\Onepiece\1001_image\', 'HR_onepiece_train_000' , num2str(index) );
  elseif index < 100
    image_name=strcat('D:\Onepiece\1001_image\', 'HR_onepiece_train_00' , num2str(index) );
  elseif index < 1000
    image_name=strcat('D:\Onepiece\1001_image\', 'HR_onepiece_train_0' , num2str(index) );
  else
    image_name=strcat('D:\Onepiece\1001_image\', 'HR_onepiece_train_' , num2str(index) );
  end
  
  index = index + 1;
  
  image_name=strcat(image_name,'.jpg');

  I=read(video,i); %?�X?��

  imwrite(I,image_name,'jpg'); %??��

  I=[];

  end
  
end