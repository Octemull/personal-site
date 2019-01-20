---
title: "在Matlab下利用Libsvm的输出模型画SVM三维特征的二分类曲面"
date: 2019-01-19T19:18:01+08:00
lastmod: 2019-01-20T12:04:00+08:00
draft: false
tags: ["Matlab","Original"]
categories: ["Machine Learning"]
show_comments: true
---

# 前言

在做毕设的时候用到了支持向量机（SVM）做分类，当特征为3维的时候，想画一个分类面出来。因为在matlab中使用的Libsvm包，没有画三维分类面的功能，所以参考了stackoverflow上的一个问题，写了一下画三维分类曲面的程序。

# 所用软件

1. Matlab R2017b
2. Libsvm-3.22 （安装在matlab的toolbox中）

# 所用数据

| 变量名 | 说明 |
| --- | --- |
| model | 用Libsvm建模输出的模型  |
| train_data | 标准化后的训练数据，每一行是一个样本，每一列是一个特征 |
| train_target | 样本标记，0-1向量 |

# 数据示例

### 说明

1. `negative`样本在前，`positive`样本在后
2. `negative`样本标记`0`，`positive`样本标记`1`
3. `train_data`已经标准化到`-1`、`1`之间（标准化后方便svm的训练和曲面的展示）
4. 训练svm的核函数为`RBF`，若训练时使用了其他的核函数，则需要修改`funRBF`

### train_data

![](https://i.loli.net/2019/01/19/5c4319e8359e0.jpg)

### train_target
![](https://i.loli.net/2019/01/19/5c431a19aa4a3.jpg)

# 训练模型

```matlab
model = svmtrain(train_target,train_data, ['-t 2 -c 100 -g', num2str(1/3),' -b 1 -q']); 
```

# 代码

### Main Code | 主代码

```matlab
close all
clear
clc

tic % 开始计时

%% load data 加载数据
load('model','train_data','train_target') 

Xdata_scaled = train_data;
group = train_target;


%% code 
GN3Dplot(Xdata_scaled,sum(group==0),sum(group==1),0,0,0,0); %plot samples 画三维图
xlabel('F1');ylabel('F3'); zlabel('F6');
ylim([-1 0]);
yticks([-1 -0.75 -0.5 -0.25 0])
yticklabels({'-1','-0.5','0','0.5','1'})
set(gca,'fontsize',14)
box on

k = 30; %grid density 网格密度，越大越慢
cubeXMin = min(Xdata_scaled(:,1));
cubeYMin = min(Xdata_scaled(:,2));
cubeZMin = min(Xdata_scaled(:,3));

cubeXMax = max(Xdata_scaled(:,1));
cubeYMax = max(Xdata_scaled(:,2));
cubeZMax = max(Xdata_scaled(:,3));
stepx = (cubeXMax-cubeXMin)/(k-1);
stepy = (cubeYMax-cubeYMin)/(k-1);
stepz = (cubeZMax-cubeZMin)/(k-1);
[x, y, z] = meshgrid(cubeXMin:stepx:cubeXMax,cubeYMin:stepy:cubeYMax,cubeZMin:stepz:cubeZMax);
mm = size(x);
x = x(:);
y = y(:);
z = z(:);
f = funRBF([x y z],model); %kernel function 核函数，此处为RBF

x0 = reshape(x, mm);
y0 = reshape(y, mm);
z0 = reshape(z, mm);
v0 = reshape(f, mm);

[faces,verts,colors] = isosurface(x0, y0, z0, v0, 0, x0);

grid on
box on

%% plot 画图
% colorful grid 彩色网格
p = patch('Faces',faces,'Vertices',verts,'facecolor','w','edgecolor','flat','CData',verts(:,3), 'FaceAlpha', 0.5)

% black grid 黑色网格
% p = patch('Faces',faces,'Vertices',verts,'facecolor','w','CData',verts(:,3),'FaceAlpha', 0.5)

% black plane with alpha 黑色曲面，可通过alpha调节透明度
% p = patch('Vertices', verts, 'Faces', faces, 'FaceColor','k','Edgecolor', 'none', 'FaceAlpha', 0.5);

set(gca,'fontsize',14) %设置图像中的字体大小
hold off

toc %结束计时
```

对isosurface的一些说明：

**patch**

重要的命令：

```matlab
'FaceVertexCData',colors
```

* 'edgecolor'网格线：若要画出彩色的网格线（’interp'），需要加入’FaceVertexCData’项，否则’edgecolor’选项只能使用’none'

![](https://i.loli.net/2019/01/20/5c445fc1bdf42.jpg)

* 'FaceColor’曲面颜色：若要画出彩色的网格线（’interp'），需要加入’FaceVertexCData’项，否则’FaceColor’选项只能使用’none'

![](https://i.loli.net/2019/01/20/5c445fd70bb82.jpg)

或

![](https://i.loli.net/2019/01/20/5c445fe337ede.jpg)

**'FaceAlpha' — Face transparency 曲面透明度**

* [0, 1]间的常数：固定的透明度，1为不透明，0为全透明
* ‘flat’: 根据FaceVertexAlphaData确定不同的透明度
* 'interp’ : 根据FaceVertexAlphaData，用差值法确定不同的透明度

**相关命令: view(a, b)**

* 作用：设置图像观察方向
* 一般b设置为30就可以，a可以根据需要调整。

### RBF kernel function | 径向基核函数

```matlab
function y = funRBF(x,model)
% x: the vector
% SVs: support vectors

gamma = model.Parameters(4);
SVs = model.SVs;
sv_coef = model.sv_coef;
b = -model.rho;

RBF = @(u,v)( exp(-gamma.*sum( (u-v).^2) ) );

for i = 1:size(x,1)
    y(i) = 0;
    for j = 1:size(SVs,1)
        u = SVs(j,:);
        y(i) = y(i) + sv_coef(j)*RBF(u,x(i,:));
    end
    y(i) = y(i) + b;
end
```

### GN3Dplot | 三维散点图的绘制

```matlab
function GN3Dplot(x,nNeg,nPos,xp,yp,zp,textFlag)
% BZ: if standardized
% function 
if size(x,2) < 3
	error('dim(x) < 3!!!');
end

n = nPos+nNeg;
target = [zeros(nNeg,1);ones(nPos,1)];

% plot
figure; %三维作图
for i = 1:n
    if target(i) == 1
        h1 = stem3(x(i,1),x(i,2),x(i,3),'filled','-r');
        hold on;
    else
%         h2 = stem3(x(i,1),x(i,2),x(i,3),'filled','-b');
        h2 = stem3(x(i,1),x(i,2),x(i,3),'-.b');
        hold on;
    end
    if textFlag == 1
        text(x(i,1)+xp,x(i,2)+yp,x(i,3)+zp,num2str(i));
    end
end
legend([h1 h2],'Positive','Negative');

xpp = (max(x(:,1))-min(x(:,1)))/10;
ypp = (max(x(:,2))-min(x(:,2)))/10;
zpp = (max(x(:,3))-min(x(:,3)))/10;
xlim([min(x(:,1)) - xpp,max(x(:,1)) + xpp]);
ylim([min(x(:,2)) - ypp,max(x(:,2)) + ypp]);
zlim([min(x(:,3)) - zpp,max(x(:,3)) + zpp]);
grid on; grid minor;
hold off;

end
```

# 图像展示

### colorful grid | 彩色网格

![](https://i.loli.net/2019/01/19/5c431e4364c09.jpg)


### black grid 黑色网格

![](https://i.loli.net/2019/01/19/5c431e84f2e96.jpg)


### black plane with alpha 黑色曲面，可通过alpha调节透明度

![](https://i.loli.net/2019/01/19/5c431ea2a92c9.jpg)


# 参考

* matlab: https://stackoverflow.com/questions/16146212/how-to-plot-a-hyper-plane-in-3d-for-the-svm-results
* python: https://www.semipol.de/2015/10/29/SVM-separating-hyperplane-3d-matplotlib.html

