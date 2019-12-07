maindir = [pwd  '\orl_faces'];
subdir = dir(maindir);
images = [];


for i = 1: length(subdir)
    if(isequal(subdir(i).name, '.')||isequal(subdir(i).name, '..')||~subdir(i).isdir)
        continue;
    end

    subdirpath = fullfile(maindir,subdir(i).name,'.pgm')
    image = dir(subdirpath);
    for j = 1:length(image)
        image_path = fullfile(maindir,subdir(i).name,image(j).name);
        img = imread(image_path)
        images = [images, double(reshape(img, 112 * 92, []))];
    end
end

mu=[];
for i = 1:40
    temp = mean(images(:,1+10*(i-1):10*i),2);
    mu = [mu,temp];
end

Sw = zeros(10304, 10304);
for i = 1:40
    for j = 1:10
        Sw = Sw + (images(:,10*(i-1)+j)-mu(:,i))*(images(:,10*(i-1)+j)-mu(:,i))';
    end
end

Sb = zeros(10304, 10304);
for i = 1:40
    Sb = Sb + (mu(:, i) - mean(mu, 2)) *(mu(:, i) - mean(mu, 2))';
end
Sb = Sb.*10;

[U, S, VT] = svd(Sw);
S = S^(-1/2);
ssw = U * S * VT;

[U, S, VT] = svd(Sb);
sb = S ^ (1 / 2) * U';

% sb = Sb(:, 1: 30);
[U, S, VT] = svd(sb'*ssw);
test = VT(1:30,:) *images;
test1 = VT(1:30,:)'*test;
imshow(uint8(reshape(test1(:, 1) + mu(:, 1), 112, 92)))
    