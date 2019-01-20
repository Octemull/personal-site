---
title: "Jupyter notebook使用技巧 + nbextentions"
date: 2018-10-30T22:14:46+08:00
lastmod: 2019-01-16T22:14:46+08:00
draft: false
show_comments: true
keywords: []
description: ""
tags: ["Tools"]
categories: ["Python"]
---

# Tab！

1. 单按Tab：代码自动补全
2. Shift + Tab：查看帮助


# 输出矢量图！

```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
```

上面的最后一行指定了输出的格式是 svg，这样浏览器就能为你渲染矢量图了。

savefig 只要指定文件名后缀是 `.pdf` 或者 `.eps` 就能生成能方便地插入 latex 的图片了！

```python
plt.savefig('tmp.pdf', bbox_inches='tight')
plt.show()
```

# magic命令！ 

> 参考：https://blog.csdn.net/tianjie5768/article/details/80164142

Magic单元分为两种，一种是line magics，另外一种cell magics。Line magic是通过在前面加`%`，表示magic只在本行有效。Cell magic是通过在前面加`%%`，表示magic在整个cell单元有效。

🌰e.g. 
下图中使用`%%bash`，产生了linux下的shell环境（window下不支持，不过可以使用`%%cmd`），这样就可以运行`pwd`和`ls`命令了。

![](https://i.loli.net/2019/01/20/5c448343f2a0f.jpg)


1、输入`%lsmagic`，可以显示所有magic命令。

2、其中一些比较常用的magic：

（1）在jupyter内打印图片
```python
%matplotline inline
```

（2）将本地py文件代码导入进来到当前单元中

```python
%load
```

🌰e.g. 
```python
%load test.py
```

（3）运行本地代码
```python
%run
```
利用这个magic，我们可以把一些头文件，基本设置，共同函数写在不同的notebook内，用的时候运行一下就可以了。

🌰e.g. 
将公共的函数写在`common_import.ipynb`中，一些导入函数的配置文件存在`utils.ipynb`中，需要的时候使用`%run`直接运行一下，就可以把公共函数和环境配置好了。这样可以将代码写成不同的模块，而不是全部写进一个notebook。

![](https://i.loli.net/2019/01/20/5c44839db1373.jpg)

# 配置文件命令

便捷获取配置文件所在路径的命令
```python
jupyter notebook --generate-config
```

注意： 这条命令虽然可以用于查看配置文件所在的路径，但主要用途是是否将这个路径下的配置文件替换为默认配置文件。 如果你是第一次查询，那么或许不会出现下图的提示；若文件已经存在或被修改，使用这个命令之后会出现询问`“Overwrite /Users/raxxie/.jupyter/jupyter_notebook_config.py with default config? [y/N]”`，即“用默认配置文件覆盖此路径下的文件吗？”，如果按`y`，则完成覆盖，那么之前所做的修改都将失效；如果只是为了查询路径，那么一定要输入`N`。

常规的情况下，Windows和Linux/macOS的配置文件所在路径和配置文件名如下所述：
* Windows系统的配置文件路径：`C:\Users\<user_name>\.jupyter\`
* Linux/macOS系统的配置文件路径：`/Users/<user_name>/.jupyter/` 或 `~/.jupyter/`
* 配置文件名：`jupyter_notebook_config.py`

# nbextensions ! 最强插件集合 ！

> 参考：https://ndres.me/post/best-jupyter-notebook-extensions/
> Extensions说明: https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions.html

## Extensions管理

Notebook extensions are plug-ins that you can easily add to your Jupyter notebooks. The best way to install them is to use [Jupyter NbExtensions Configurator](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator). It will add a tab to let you enable/disable extensions.

**先确认juppyter notebook已关闭， 否则容易安装失败(只有extension页面，没有具体的extensions)**

打开终端，

1.安装
```
pip install jupyter_nbextensions_configurator
```

2.启用
```
jupyter contrib nbextension install --userjupyter nbextensions_configurator enable --user
```

3.打开jupyter notebook即可使用. 在想要开启的插件前打钩即可。

![](https://i.loli.net/2019/01/20/5c44845d7e299.jpg)


## 常用extension推荐

1. **Code prettify**: 代码美化，需要安装yapf包
2. **Collapsible headings**: Very useful when dealing with large notebooks, collapsible headings allow you to collapse some parts of the notebooks.
3. **Notify**: For long running task, the notify extension sends a notification when the notebook becomes idle.
4. **Code folding**: fold the code
5. **ExecuteTime**: show the running time for each block
6. **Scratchpad**: draft 草稿本（单击右下角出现一个有方框的小三角即可使用）
7. **Table of Contents(2)**: generate contens
8. **Variable inspector**: displays all variables in a floating window 

# 快捷键 !

① 命令模式

![55CD229B-4109-4329-8220-F9F25D98D258](https://i.loli.net/2019/01/20/5c4485adda6e2.jpg)


② 编辑模式

![](https://i.loli.net/2019/01/20/5c44859748267.jpg)


# 参考：

1. https://www.zhihu.com/question/59392251
2. https://zhuanlan.zhihu.com/p/33105153