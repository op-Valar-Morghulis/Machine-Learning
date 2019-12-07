clear all; clc
% 数据处理
f=fopen('/Users/chenjiarui/Desktop/processData.csv');
% 每一行
wine=textscan(f,'%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');
fclose(f);
% data=4898*12 (数据规格）
for i=1:12
    data(:,i)=wine{:,i};
end
[m,n]=size(data);

%主程序
%交叉验证 
%10次10折交叉验证划分数据集
cov_iter=1;%交叉验证迭代次数
k=10;%划分10个子集
while cov_iter<=k
train=data;
train_index=0;
test_index=1;
for i=0:1
    s=sum(data(:,n)==i);
    total=round(s/k);
    sel=randperm(s,total);
    test(test_index:test_index+total-1,:)=data(train_index+sel,:);
    train(train_index+sel,:)=[];
    train_index=train_index+s-total;
    test_index=test_index+total;
end
P=0.9;
tree=decision_tree(train,P);
[right,error]=testtree(tree,test,0,0);
accuracy(cov_iter)=right/(right+error);
cov_iter=cov_iter+1;
end
display(strcat('测试集平均准确率为',num2str(mean(accuracy)*100),'%'));


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


%计算熵的函数
function [Ent_A,Ent_B] = Ent(data,lamda)
%data为按照某一属性排列后的数据集
%lamda记录按某一属性排序后二分的位置
c=1;   %c作为label类别，一共两个label（0和1）
[m,n]=size(data); %理论来说这里m=4898 n=12
% disp(m,n)
data1=data(1:lamda,:);
data2=data(lamda+1:end,:);
%将数据分为两个组
Ent_A=0; Ent_B=0;
%计算熵
for i=0:c
    order=i+1;
    data1_p(order)=sum(data1(:,n)==i)/lamda;
    if data1_p(order)~=0
        Ent_A=Ent_A-data1_p(order)*log2(data1_p(order));
    end
    data2_p(order)=sum(data2(:,n)==i)/(m-lamda);
    if data2_p(order)~=0
        Ent_B=Ent_B-data2_p(order)*log2(data2_p(order));
    end
end
end


%寻找最优化分属性 attribute
%最优化分数值 value
function [attribute,value,lamda]=pre(data)
total_attribute=11;  %属性个数
Ent_parent=0;  %父节点熵
[m,n]=size(data);
% 计算对于二分类任务父节点的总熵值
for j=0:1
    order=j+1;
    data_p(order)=sum(data(:,n)==j)/m;
    if data_p(order)~=0
        Ent_parent=Ent_parent-data_p(order)*log2(data_p(order));
    end
end
%计算最优化分属性
min_Ent=inf;
for i=1:total_attribute
    data=sortrows(data,i);
    for j=1:m-1
        if data(j,n)~=data(j+1,n)
            [Ent_A,Ent_B]=Ent(data,j);
            Ent_pre=Ent_A*j/m+Ent_B*(m-j)/m;
            if Ent_pre<min_Ent
                min_Ent=Ent_pre;
                %记录此时的划分数值和属性
                attribute=i;
                value=data(j,i);
                lamda=j;
            end
        end
    end
    %信息增益
    gain=Ent_parent-min_Ent;
    if gain<0
        disp('算法有误，请重新设计');
        return 
    end
end
end

%建立决策树
%P控制精确度，防止分支过多，即过拟合现象
function tree=decision_tree(data,P)
[m,n]=size(data);
tree=struct('value','null','left','null','right','null');
[attribute,value,lamda]=pre(data);
data=sortrows(data,attribute);
tree.value=[attribute,value];
tree.left=data(1:lamda,:);
tree.right=data(lamda+1:end,:);
%终止构建条件
for i=0:1
    order= i+1;
    left_label(order)=sum(tree.left(:,n)==i);
    right_label(order)=sum(tree.right(:,n)==i);
end
    [num,max_label]=max(left_label);
    if num~=lamda&&(num/lamda)<P
        tree.left=decision_tree(tree.left,P);
    else
        tree.left=[max_label num];
    end
    [num,max_label]=max(right_label);
    if num~=(m-lamda)&&(num/(m-lamda))<P
        tree.right=decision_tree(tree.right,P);
    else 
        tree.right=[max_label num];
    end
end

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
