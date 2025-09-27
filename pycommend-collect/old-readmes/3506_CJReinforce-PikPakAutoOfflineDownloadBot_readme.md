# 更新日志
## 2024-12-5
- 新增功能：支持指定离线下载至PikPak某个目录
- 发送 /path 命令获取使用帮助
- 同时支持快速设置临时下载目录：发送/p /movie magnet 就可以临时将离线下载路径修改为 /movie
- 本次更新请在config中建立新参数PIKPAK_OFFLINE_PATH = "None"

# 功能

自动PikPak离线下载+aria2下载+释放网盘空间的TG机器人

# 用途

得益于PikPak网盘与迅雷之间千丝万缕的联系，PikPak网盘的离线下载功能常常能做到秒离线。其服务器上资源之多，使其被戏称为”迅雷新加坡分雷“。对于已经下载不动的老磁力，不妨试试PikPak的离线下载，或许会有惊喜。

本项目实现了一个可以一键将磁力链接经pikpak离线后再下载到本地并删除对应网盘文件的tg机器人。只需简单配置，即可做到：磁力不担心，来去无痕迹。我只为磁力而来，不沾染一片尘埃。

# 重要提示

**不建议将存储重要文件的PikPak账号用于本项目！**

因为部分命令删除文件的机制较强劲，容易在使用中操作不慎导致误删。

# 本地部署

将项目文件下载到本地同一目录下。

安装依赖：

```shell
pip install -r requirements.txt
```

配置`config.py`文件信息：

```python
# TG机器人的令牌，tg找@BotFather创建机器人即可获取
TOKEN = 'token'
# TG用户ID，指定用户才能使用机器人
ADMIN_IDS = ['12345678']
# pikpak账号，可以为手机号、邮箱，支持任意多账号
USER = ["example_user1", "example_user2"]
# 账号对应的密码，注意与账号顺序对应！！！
PASSWORD = ["example_password1", "example_password2"]
# 自动删除配置，未配置默认开启自动删除，留空即可
# AUTO_DELETE = {"example_user1": "True", "example_user2": "False"}
AUTO_DELETE = {}
# 以下分别为aria2 RPC的协议（http/https）、host、端口、密钥
ARIA2_HTTPS = False
ARIA2_HOST = "example.aria2.host"
ARIA2_PORT = "port"
ARIA2_SECRET = "secret"
# aria2下载根目录
ARIA2_DOWNLOAD_PATH = "/mnt/sda1/aria2/pikpak"
# 可以自定义TG API，也可以保持默认
TG_API_URL = 'https://api.telegram.org'
# 自定义Pikpak离线下载路径
PIKPAK_OFFLINE_PATH = "None"
```

最后：

```shell
python pikpakTgBot.py
```

这样你的机器人就上线工作啦！当然最好还是使用如`pm2`等进程守护工具在后台守护运行。

# Docker Compose 部署

```shell
git clone 本项目
# 编辑`config.py`文件，配置信息如上所述。
# 构建镜像
docker-compose build
# 启动容器，后台运行
docker-compose up -d
```

其他参考命令：

```shell
# 查看容器状态
docker-compose ps
# 停止、启动、重启容器
docker-compose stop | start | restart
# 停止容器并删除容器
docker-compose down
# 启动容器，后台运行
docker-compose up -d
# 查看日志信息
docker logs pikpakbot
```

# Docker 部署

1. 将项目文件下载到本地或者直接`git clone`本项目。
2. 编辑`config.py`文件，配置信息如上所述。
3. 目录结构如下：
```shell
PikPakAutoOfflineDownloadBot
├── Dockerfile
├── README.md
├── __init__.py
├── config.py
├── docker-compose.yml
├── pikpakTgBot.py
└── requirements.txt
```

4.制作docker镜像，运行容器

```shell
cd /root/PikPakAutoOfflineDownloadBot
```

```shell
docker build . --tag pikpakbot
```

```shell
docker run \
  --name=pikpakbot \
  --restart=always \
  -d \
  -v /root/PikPakAutoOfflineDownloadBot:/code \
  pikpakbot
```

6.运行文件采用挂载方式，如果需要修改配置，可以直接修改`/root/PikPakAutoOfflineDownloadBot`下的文件，然后重启容器即可。

# 使用

机器人监听的命令如下：

| 命令                                                                               | 含义        | 用法                                 | 备注                                                                                                                                               |
|----------------------------------------------------------------------------------|-----------|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `/start`                                                                         | 获取帮助信息    | `/start`                           | 无                                                                                                                                                |
| `/help`                                                                          | 获取帮助信息    | `/help`                            | 无                                                                                                                                                |
|                                                                                  |           |                                    |                                                                                                                                                  |
| `/p`                                                                             | 一键下载磁力到本地 | `/p magnet1 [magnet2] [...]`       | 支持多个磁力链接；直接发送磁力链接也能识别；支持pikpak能够解析的普通链接，如Twitter视频、ed2k链接等<br/> 新功能：支持直接设置临时下载路径，比如 /downloads  magnet1 将直接离线到/downloads目录，注意：只支持在开头设置临时路径（绝对路径） |
| `/clean`                                                                         | 清空指定账号的网盘 | `/clean account1 [account2] [...]` | `/clean all`清空所有账号网盘                                                                                                                             |
| `/account`                                                                       | 管理账号      | `/account l/a/d/n [parameters]`    | 向机器人发送`/account`获取详情                                                                                                                             |
| `/path`                                                                          | 管理离线路径    | `/path info/default/[parameters]`  | 向机器人发送`/path`获取详情                                                                                                                                |

**`/clean`命令清空文件无法找回！请慎用！**

部分命令使用情况如下图所示：

| ![`/pikpak`命令截图](https://s3.bmp.ovh/imgs/2022/06/08/8d3fdd294c98a871.png) | ![`/pikpak`命令](https://s3.bmp.ovh/imgs/2022/06/08/7e2eec33f35d17e2.png) |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| ![`/pikpak`失败案例](https://s3.bmp.ovh/imgs/2022/06/08/812b258e14273fe2.png) | ![`/clean`命令](https://s3.bmp.ovh/imgs/2022/06/08/05049c4f5a73f29f.png)  |







# 注意事项

## 程序相关

- pikpak离线下载时可能返回`xx not saved successfully!`的信息，
  原因为pikpak
  默认不离线广告文件
- pikpak离线下载可能会长时间卡在0进度，这表明pikpak服务器没有此资源，所以下不动。此时程序会停止下载此磁链
- 所有待下载的磁力会按顺序逐个下载，只有完成上一个磁力从离线到下载至本地再释放网盘空间的全部过程，才会继续处理下一个磁力。这是为了避免出现网盘空间不够用的情况
- `/p`命令不会阻塞进程，意味着可以在正在下载上一个磁力的过程中，继续添加磁力，但是依然会排队等待下载
- `/p`命令可能存在部分文件下载失败的情况，tg机器人会发送消息给出解决方案，也欢迎带日志反馈失败的情况
- `/clean`命令会阻塞进程，这是为了避免出现一边下一边删的情况
- tg机器人发送消息较少且较简洁，但程序的日志内容较为详尽，如有bug请带日志反馈

## 其他

- 本项目没有任何破解行为，因此如普通用户6G空间限制、每天三次离线机会等限制均存在
- 获取账号功能已失效，如有需要请自行注册获取


# 参考

- [666wcy/pikpakdown](https://github.com/666wcy/pikpakdown)
- [mumuchenchen/pikpak](https://github.com/mumuchenchen/pikpak)
- [Quan666/PikPakAPI](https://github.com/Quan666/PikPakAPI)
