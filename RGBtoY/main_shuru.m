clc;
clear all;
for num=1:1
  I_ir=(imread(strcat('D:\MEF\test_imgs\benchmark100\over\',num2str(num),'.png')));  
  [Y,Cb,Cr]=RGB2YCbCr(I_ir); 
%   fprintf(Y)
  imwrite(Y, strcat('D:\MEF\FS\',num2str(num),'.png'));
%   imwrite(v1,strcat('',num2str(num),'.bmp'));
%   imwrite(v2,strcat('',num2str(num),'.bmp'));  
end