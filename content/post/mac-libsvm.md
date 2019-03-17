---
title: "Mac下matlab2014b安装libsvm"
date: 2018-03-22T20:56:49+08:00
lastmod: 2019-01-20T20:56:49+08:00
draft: false
show_comments: true
keywords: []
description: ""
tags: ["Mac"]
categories: ["Machine Learning"]
---

# 环境&软件说明

1. 系统：macOS High Sierra 10.13.3
2. matlab版本：matlab2014b
3. xcode版本：Xcode9.1， SDK版本10.13 （必须要有xcode才行）
4. libsvm版本：libsvm3.22


# 下载libsvm

1）在[libsvm主页](https://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html#matlab)下载最新的libsvm

![](https://i.loli.net/2019/01/20/5c44708992e12.jpg)

2）直接在Downloads下解压

3）将解压后的文件夹复制到`/Applications/MATLAB_R2014b.app/toolbox/`下
（P.S. 从Finder中的Application里找到Matlab，右键显示包内容）

![](https://i.loli.net/2019/01/20/5c4470d544354.jpg)


# 下载xcode、安装command line tool

1）安装command line tools：打开终端(Terminal)，输入

```terminal
xcode-select --install
```

 然后点击安装，等待下载安装即可。

2）确认xcode的SDK版本

从Finder进入
`/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs` ，

看到SDK版本为`10.13`（Xcode9.1）

3）修改xml文件**【MATLAB2017b可跳过这一步直接编译】**

> 参考：https://blog.csdn.net/wukong1981/article/details/72805084）
因为matlab2014b不支持（自动识别）10.13版本的SDK，所以要在XML里添加几行

- 打开matlab，在command window中输入

```matlab
edit ([matlabroot '/bin/maci64/mexopts/clang_maci64.xml'])
```

查找`"10.9"`关键词，得到例如：
```xml
<dirExists name="
/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk"/><cmdReturnsname="find/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk"/><cmdReturnsname="find
-name MacOSX10.9.sdk" />
```

在下面依次的加入`10.12`，`10.13`加好之后应该是这个样子的
```xml
</XCODE_AGREED_VERSION>
        <ISYSROOT>
            <and>
                <cmdReturns name="xcode-select -print-path"/>
                <or>
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.9.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.10.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.11.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.12.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.13.sdk" />
                </or>
            </and>
        </ISYSROOT>
        <SDKVER>
            <and>
                <and>
                    <cmdReturns name="xcode-select -print-path"/>
                    <or>
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.9.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.10.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.11.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.12.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.13.sdk" />
                    </or>
                </and>
                <cmdReturns name="echo $$ | rev | cut -c1-10 | rev | egrep -oh '[0-9]+\.[0-9]+'" />
            </and>
        </SDKVER>
```

**注意有 `<ISYSROOT>`、 `<SDKVER>` 两个地方需要修改**

同样，为了编译C++文件，需要对下面文件做同样的处理

```matlab
edit ([matlabroot '/bin/maci64/mexopts/clang++_maci64.xml'])
```

结果如下

```xml
</XCODE_AGREED_VERSION>
        <ISYSROOT>
            <and>
                <cmdReturns name="xcode-select -print-path"/>
                <or>
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk" />
                    <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.9.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.10.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.11.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.12.sdk" />
                    <cmdReturns name="find $$ -name MacOSX10.13.sdk" />
                </or>
            </and>
        </ISYSROOT>
        <SDKVER>
            <and>
                <and>
                    <cmdReturns name="xcode-select -print-path"/>
                    <or>
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk" />
                        <dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.9.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.10.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.11.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.12.sdk" />
                        <cmdReturns name="find $$ -name MacOSX10.13.sdk" />
                    </or>
                </and>
                <cmdReturns name="echo $$ | rev | cut -c1-10 | rev | egrep -oh '[0-9]+\.[0-9]+'" />
            </and>
        </SDKVER>
```

**改完后一定要重启一下matlab**。

# 编译安装libsvm

- 导向到`matlab根目录`，进入`libsvm`的`matlab`文件所在文件夹

> 参考：https://blog.csdn.net/u013515273/article/details/51276184）

```matlab
cd(matlabroot)
```
 运行

```matlab
mex -setup 
```

和 

```matlab
mex -setup C++
```

- 再重新导向到`libsvm`所在目录的`matlab文件夹`，运行`make`，

```matlab
cd toolbox/libsvm-3.22/matlab
make
```

该步骤将C++文件编译成matlab下可以运行的文件，编译成功后文件夹下会生成`.mexmaci64`文件。

编译成功后就可以使用`libsvm`下的`svmtrain`，`svmpredict`等命令了。（注意matlab也自带了一个svmtrain命令，为了保证使用的是libsvm的svmtrain，需要设置当前目录为`${libsvm}/matlab`。）

**推荐：不设置当前目录，引入libsvm工具包。操作如下：**

- 引入工具包：

在HOME标签页上点击Set Path；左侧点Add With Subfolders，把`libsvm`中`matlab`文件夹加进去，保存就好了。 

![](https://i.loli.net/2019/01/20/5c447f7d833c8.jpg)

# 测试

首先进入heart_scale文件目录，就是libsvm的目录；

![](https://i.loli.net/2019/01/20/5c447fae545b2.jpg)

```matlab
cd(matlabroot)
cd toolbox/libsvm-3.22/
```

然后依次输入下面的代码：

```matlab
[heart_scale_label,heart_scale_inst]=libsvmread('heart_scale'); 

model = svmtrain(heart_scale_label,heart_scale_inst, '-c 1 -g 0.07'); 

[predict_label, accuracy, dec_values] =svmpredict(heart_scale_label, heart_scale_inst, model);
```

结果为

![](https://i.loli.net/2019/01/20/5c447fce66e54.jpg)


出现 
```matlab
Accuracy = 86.6667% (234/270) (classification)
```
测试成功！


# 后记：关于重命名

一般情况下还是不要重命名了，要用MATLAB自带的svmtrain等函数，就把之前添加的路径`remove`就好。

重命名可能出现的问题：

1. make后重命名maci64的svmtrain为libsvmtrain，不能正常引用；
2. 若make前就重命名，则编译（make）过程会报错。