# Lazy-Balancer

项目起源于好哥们需要一个 7 层负载均衡器，无奈商业负载均衡器成本高昂，操作复杂。又没有特别喜欢（好看，好用）的开源产品，作为一名大 Ops 怎么能没有办法？正好最近在看 Django 框架，尝试自己给 Nginx 画皮，项目诞生！非专业开发，代码凑合看吧。

> * 项目基于 [Django](https://www.djangoproject.com/) + [AdminLTE](https://www.almsaeedstudio.com/) 构建，在 Ubuntu 18.04 上测试通过；为了保证良好的兼容性，请使用 Chrome 浏览器。
> * 为了后续扩展方便，请大家使用 [Tengine](http://tengine.taobao.org/) 替代 Nginx 服务

## 项目地址

- GITHUB - https://github.com/v55448330/lazy-balancer
- 码云 - http://git.oschina.net/v55448330/lazy-balancer
- OSCHINA - http://www.oschina.net/p/nginx-balancer

## 更新（2024-10-31）

* 新增 全局配置中支持自定义日志格式
* 优化 新增部分功能提示信息
* 优化 适配移动端浏览器样式

> 本次更新涉及数据库变动，无法直接升级，请备份配置后，删除数据库并重新还原配置

## 更新（2024-07-18）

* 优化 修改 DNS 解析超时时间为 5 秒，防止域名解析慢导致的 Nginx 启动失败
* 优化 配置同步逻辑，从节点同步前会检测主节点配置状态，如果配置异常则直接同步失败
* 优化 Docker 启动命令默认限制容器日志大小，请按需修改
* 优化 新增默认数据库，新环境不再需要执行初始化数据库操作
  
  > 如果升级，请备份您的数据库或配置文件，防止覆盖或丢失
* 新增 Dashboard 页面增加 Nginx 手动开关

  > 操作接口会占用本地 9001，更新时请确保没有规则使用该端口

* 新增 同步状态及配置查询 API，请查阅 API 文档
* 新增 配置同步失败关闭 Nginx 功能（可选），防止多节点同步异常导致的服务异常
  
  > 请自行监控并使用前端 LB 执行健康检测
* 修复 Vagrant 部署脚本，快速搭建本地开发环境

## 更新（2024-01-23）

* 新增 Dashboard 页面 5 秒自动刷新所有数据
* 新增 [nginx-module-vts](https://github.com/vozlt/nginx-module-vts) 模块，实现更完善的流量监测能力
  * 新增 Dashboard 页面 Nginx 启动时间显示
    
    > API 中同步增加字段，单位为 ms，可以自行转换
  * 新增 Dashboard 页面中 TCP 流量监测功能
  * 新增 Dashboard 页面中流量统计 MB/GB 单位自适应显示，单位超过 GB 会加粗显示
  * 新增 Prometheus 格式流量监测接口
    
    > 该服务会占用 `9191/tcp` 端口
    > 可以在系统设置中打开 `公开指标接口` 功能，以实现外部监控，该功能可能造成隐私泄露等安全风险，建议使用 Telegraf 等方案从本地收集数据
    > 
    > 上游服务器健康状态（HTTP/TCP）`<BASE_URL>:9191/up_status?format=[prometheus|json|html]`
    > 
    > 流量统计（HTTP）`<BASE_URL>:9191/req_status_http/format/[prometheus|json|html]/`
    > 
    > 流量统计（TCP）`<BASE_URL>:9191/req_status_tcp/format/[prometheus|json|html]/`
* 优化 因插件功能冲突，动态域名解析功能，由原 [ngx_upstream_jdomain](https://github.com/nicholaschiasson/ngx_upstream_jdomain) 模块更换为 Tengine 自带 [ngx_http_upstream_dynamic](https://tengine.taobao.org/document_cn/http_upstream_dynamic_cn.html) 模块
  
  > 因 `ngx_http_upstream_dynamic` 模块和主动健康检测模块及负载均衡算法实现冲突，开启动态域名解析功能后，需要由 DNS 实现负载均衡及健康检测
* 优化 暂时精简 LuaJIT 环境
* 优化 引入 [jemalloc](https://jemalloc.net/) 进行内存管理优化
* 优化 默认使用 [VNSWRR](https://tengine.taobao.org/document_cn/ngx_http_upstream_vnswrr_module_cn.html) 算法替代 NGINX 负载均衡算法

## 更新（2023-12-03）

* 更新 Python 到 3.9
* 更新 Tengine 到 3.1.0
* 更新 LuaJIT 到 20231006
* 更新 Alpine 到 3.18.4
* 更新 部分 Python 依项
* 优化 在所有保存配置操作前均执行 nginx -t 验证已存在配置，如果异常将不会执行配置渲染及后续操作
* 优化 状态页面拆分配置和进程状态，现在配置异常不会影响 Nginx 状态，将会独立显示 “配置异常” 状态
* 优化 服务启动流程，服务启动时将重新生成并应用配置文件，防止错误配置导致的启动失败
* 优化 修改状态/删除规则时，将允许只保存，不应用配置，以防止现有多条配置失效导致的 nginx 启动失败
* 优化 导入/保存/重新生成等操作将临时禁用按钮，防止重复点击导致的错误
* 优化 修改了规则保存逻辑，现在只有在规则检查失败后才会重新渲染配置，在规则数量较多时，极大提高保存速度
* 优化 默认关闭错误页服务器详细信息显示
* 新增 “重新应用配置” 功能，可以手动重新渲染或重载 Nginx 配置，默认重新渲染，选择取消后可选择仅重载配置
* 新增 stream 模块中 proxy_timeout 1800s 固定配置项
* 新增 check_shm_size 32M 固定配置项
* 新增 规则列表分页长度配置，默认 10 条，可配置 10-100 条分页
* 新增 测试支持 ARM 架构，pull 镜像可以使用 `--platform linux/arm64` 参数
* 新增 HTTP 类型规则后端节点域名动态检测 [ngx_upstream_jdomain](https://github.com/nicholaschiasson/ngx_upstream_jdomain) 实现，防止 upstream 域名 IP 变动，仅支持 HTTP 协议
* 新增 更换主动健康检测模块 [ngx_healthcheck_module](https://github.com/zhouchangxun/ngx_healthcheck_module)，以解决动态域名模块兼容性问题，并增加 TCP 规则的后端节点检测功能
* 修复 在 SSL 状态下打开后端域名开关不生效的问题
* 修复 部分情况下配置错误导入失败无法回滚的 Bug，优化了导入逻辑，略微提升了导入速度
* 修复 其他交互 Bug

## 更新（2021-06-16）

* IPv6 监听支持
* 更新 Tengine 至 2.3.3

## 更新（2020-01-21）

* 从该版本开始，将尝试部分功能 API 化，更多 API 文档见 `/api/docs`
* 尝试将 Python 更新至 Python3
* 修复 TCP 模式下端口占用检测无效的问题

## 更新（2019-11-22）

* 新增 TCP 负载均衡支持
* 新增配置同步功能
* 支持后端服务器为 HTTPS 协议，当后端为 HTTPS 协议时，HTTP 健康检测将使用发送 SSL Hello 包的方式
* 支持域名后端，配置为域名后端时禁用节点配置
* 新增 HTTP/2，Gzip 等配置
* 增加 Docker 支持
* 去除原 iptables 防火墙管理功能
* 当协议为 HTTP/HTTPS 时，允许用户自定义 Server 级别 Nginx 配置
* 当协议为 HTTP/HTTPS 时，可以在列表页预览后端节点状态
* 当协议为 HTTP/HTTPS 时，允许用户自定义后端节点域名，当未定义时，转发用户输入的域名
* 当协议为 HTTPS 时，可以在列表页预览证书过期状态，及获取证书信息
* 允许后端节点为域名格式
* 增加 HTTP/80，HTTPS/443 的默认规则，禁止直接 IP 访问（返回444），证书路径在 `/etc/nginx/default.*`，可自行更换
* 新增允许非标准 HTTP Header 转发（如下划线_）
* 修复其他 Bug

## 更新

* 将 Nginx 更换为 Tengine 以提供更灵活的功能支持以及性能提升
* 新增 HTTP 状态码方式检测后端服务器，默认 TCP 方式
* 新增 HTTP 状态码方式支持查看后端服务器状态
* 修复因前方有防火墙导致无法获取后端服务器状态
* 修复因主机头导致后端服务器探测失败
* 新增自定义管理员用户
* 新增配置通过文件备份和还原
* 新增实时查看访问日志和错误日志
* 新增实时请求统计
* 更新 Vagrantfile
* 修复其他 Bug

## 功能

* Nginx 可视化配置
* Nginx 负载均衡（反向代理）配置
* Nginx 证书支持
* 系统状态监测
* 支持 TCP 被动后端节点宕机检测
* 支持 HTTP 主动后端节点宕机检测
* 日志实时查询
* 请求统计

## 运行

### 容器

* 编译镜像
  
  ```
  nerdctl build --platform=arm64,amd64 -t <lazy-balancer>:<v1.4.0beta> .
  ```
  
  > 也可以 DockerHub `https://hub.docker.com/r/v55448330/lazy-balancer`

* 启动命令
  
  ```
  docker run -d --restart=always --net=host --name=lazy_balancer \
    --log-opt max-size=500m \
    --log-opt max-file=3 \
    -v <db_dir>:/app/lazy_balancer/db \
    -v <log_dir>:/var/log/nginx \
    <lazy-balancer>:<v1.4.0beta> or v55448330/lazy-balancer:latest
  ```
  
  ### 主机

* 部署

> 部署方式参照 `deploy.sh` 脚本

* 启动服务
  
  ```
  supervisord -c /app/lazy_balancer/service/supervisord_docker.conf
  ```

  or
    ```
    supervisorctl start webui
    supervisorctl start nginx
    ```
  
* 登录系统
  ```
  http://[IP]:8000/  
  ```
  > 首次登陆会要求创建管理员用户，如需修改，可在系统配置中重置管理员用户

## 演示

### PC 端

![image](readme_img/1.jpg)
![image](readme_img/2.png)
![image](readme_img/3.png)
![image](readme_img/4.png)
![image](readme_img/5.jpg)
![image](readme_img/6.jpg)
![image](readme_img/7.jpg)
![image](readme_img/8.png)
![image](readme_img/9.png)
![image](readme_img/10.jpg)
![image](readme_img/11.jpg)
![image](readme_img/12.png)
![image](readme_img/13.jpg)
![image](readme_img/14.jpg)

### Mobile 端

![image](readme_img/m1.png)
![image](readme_img/m2.png)
![image](readme_img/m3.png)
![image](readme_img/m4.png)

## 授权
本项目由 [小宝](http://www.ichegg.org) 维护，采用 [GPLv3](http://www.gnu.org/licenses/gpl-3.0.html) 开源协议。欢迎反馈！欢迎贡献代码！
```
