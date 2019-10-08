function [X_norm,mu,sigma] = featureNormalize(X)
X_norm = X;
mu =zeros(1,size(X,2));
sigma =zeros(1,size(X,2));
m = size(X,1);
mu = mean(X);
for i =1:m
    X_norm(i,:)=X(i,:)-mu;
end
sigma =std(X);
for i = 1:m
    X_norm(i,:) = X_norm(i,:)./sigma;
end
end

