# 项目名称

URL-to-播客-to-小宇宙

中文版NotebookLM 最好用的平替

## 项目简介

把任意url转成播客，然后推送到小宇宙平台

## 体验地址

http://podlm.ai/

http://boke.tingwu.co/

## 文件结构

项目主要包含以下文件:

- `server.py`: 合成任务后端服务，长时间运行多线程执行合成任务
- `server_pro.py`: 所有功能与server.py一致，但多了小宇宙自动发布逻辑
- `api.py`: web及api等服务实现，需要长时间运行
- `task_list.json`: 用于储存所有合成记录
- `del.html`: 删除合成记录ui
- `list.html`: 所有合成记录ui
- `index.html`: 首页ui
- `resources`: 资源文件夹，包含css和js

## 安装和运行

1. 克隆或下载本项目到本地。

2. 在命令行中进入项目目录。

   ```shell
   cd podlm-public
   ```

3. 确保您的系统已安装 Python (推荐使用 Python 3.11.5 版本)，例如使用conda创建一个python环境并激活环境：

   ```shell
   conda create -n podlm-public python=3.11.5 -y

   conda activate podlm-public
   ```

4. 安装依赖

   需要[下载ffmpeg](https://ffmpeg.org/download.html)并将其配置到环境变量`Path`中，然后执行下面命令安装其他依赖：

   ```shell
   pip install -r requirements.txt
   ```

5. 配置`LLM模型`+`TTS服务信息`

   复制[config.demo.py](config.demo.py)配置到`config.py`，然后根据注释说明修改`config.py`中的配置项
   ```shell
   cp config.demo.py config.py
   ```

6. 运行以下命令启动程序:

   启动`api.py`
   ```python
   python api.py
   ```

   启动服务
   ```shell
   # 或者 server_pro.py 差异在于server_pro.py多了 小宇宙自动发布 逻辑
   python server.py 
   ```

## 访问

访问 http://127.0.0.1:8811/ 首页

访问 http://127.0.0.1:8811/list.html 所有合成记录

访问 http://127.0.0.1:8811/del.html 可以删除合成记录

## 联系方式

妙云 ceo@tingwu.co https://tingwu.co