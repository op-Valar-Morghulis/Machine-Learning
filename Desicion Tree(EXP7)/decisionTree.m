clear
clc
f=fopen('/Users/chenjiarui/Desktop/ex7Data/ex7Data.csv');
wine=textscan(f,'%f%f%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');
fclose(f);
%将标签放在最后一列
[m,n]=size(wine);
for i=1:n-1
    data(:,i)=wine{:,i+1};
end
data(:,n)=wine{:,1};




%数据可视化，准备绘树状图
function [A,iter]=print_tree(tree,A,iter,nodes)
A{iter,1}=strcat('attribute:',num2str(tree.value(1)));
A{iter,2}=strcat('阈值:',num2str(tree.value(2)));
A{iter,3}=nodes;
iter=iter+1;nodes=iter-1;
if isstruct(tree.left)
    [A,iter]=print_tree(tree.left,A,iter,nodes);
else
    A{iter,1}=strcat('label:',num2str(tree.left(1)));
    A{iter,2}=strcat('个数:',num2str(tree.left(2)));
    A{iter,3}=nodes;
    iter=iter+1;
end
if  isstruct(tree.right)
    [A,iter]=print_tree(tree.right,A,iter,nodes);
else
    A{iter,1}=strcat('label:',num2str(tree.right(1)));
    A{iter,2}=strcat('个数:',num2str(tree.right(2)));
    A{iter,3}=nodes;
    iter=iter+1;
end
end

%计算测试集正确、错误类别个数
function [right,error]=testtree(tree,test,right,error)
%right代表测试集中判断正确类别的个数
%error代表测试集中判断错误类别的个数
attribute=tree.value(1);
value=tree.value(2);
test_left=test(find(test(:,attribute)<=value),:);
test_right=test(find(test(:,attribute)>value),:);
if isstruct(tree.left)
    [right,error]=testtree(tree.left,test_left,right,error);
else
    [m,n]=size(test_left);
    for i=1:m
        if test_left(i,n)==tree.left(1)
            right=right+1;
        else
            error=error+1;
        end
    end
end
if isstruct(tree.right)
    [right,error]=testtree(tree.right,test_right,right,error);
else
    [m,n]=size(test_right);
    for i=1:m
        if test_right(i,n)==tree.right(1)
            right=right+1;
        else
            error=error+1;
        end
    end
end
end

%绘制决策树模型
A={};%A为节点描述
iter=1;%iter为迭代次数
nodes=0;%根节点序列
[A,~]=print_tree(tree,A,iter,nodes);
[m,n]=size(A);
for i=1:m
    nodes(i)=A{i,n};
    name1{1,i}=A{i,1};
    name2{1,i}=A{i,2};
end
treeplot(nodes);
[x,y]=treelayout(nodes);
for i=1:m
text(x(1,i),y(1,i),{[name1{1,i}];[name2{1,i}]},'VerticalAlignment','bottom','HorizontalAlignment','center')
end
title(strcat('葡萄酒决策树精度为',num2str(P*100),'%'));
