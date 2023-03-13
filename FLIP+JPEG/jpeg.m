function [ex_img] = jpeg(img,Q)

   A = img;
   imwrite(A ,"A.jpeg",'quality',Q);
   ex_img = imread("A.jpeg");