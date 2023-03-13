function [img] = fanzhuan(img)

init = img;
[R, C] = size(init(:,:,1));
res1 = zeros(R, C);
res2 = zeros(R, C);
res3 = zeros(R, C);

for i = 1 : R
    for j = 1 : C
        x = i;
        y = C - j + 1;
        res1(x, y) = init(i,j,1);
        res2(x, y) = init(i,j,2);
        res3(x, y) = init(i,j,3);
    end
end

img(:,:,1) = res1;
img(:,:,2) = res2;
img(:,:,3) = res3;
