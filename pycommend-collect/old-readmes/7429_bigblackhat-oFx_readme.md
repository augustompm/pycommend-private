[//]: # (# 🌖 oFx)

![ofx](https://socialify.git.ci/bigblackhat/ofx/image?description=1&font=Rokkitt&forks=1&issues=1&logo=https%3A%2F%2Fimg.icons8.com%2F%3Fsize%3D100%26id%3DF83fTNEFxxFm%26format%3Dpng%26color%3D000000&name=1&pattern=Floating%20Cogs&pulls=1&stargazers=1&theme=Light)

[//]: # ([![Python 3.x]&#40;https://img.shields.io/badge/python-3.x-yellow.svg&#41;]&#40;https://www.python.org/&#41;)

[//]: # ([![License]&#40;https://img.shields.io/badge/license-GPLv3-brown.svg&#41;]&#40;https://github.com/bigblackhat/oFx/blob/main/LICENSE&#41;)

[//]: # ([![POC_NUM]&#40;https://img.shields.io/badge/poc_num-209-orange.svg&#41;]&#40;#PocSupport&#41;)

[//]: # (![GitHub Repo stars]&#40;https://img.shields.io/github/stars/bigblackhat/ofx?color=gree&#41;)

[//]: # (![GitHub forks]&#40;https://img.shields.io/github/forks/bigblackhat/oFx?color=blue&#41;)

[//]: # (![show]&#40;img/007.jpeg&#41;)

[//]: # (笔者很喜欢海王三叉戟的造型，就画了一张作为oFx的LOGO吧)

[//]: # (2022HW专题：[点我点我]&#40;https://github.com/bigblackhat/oFx/blob/main/docs/2022HW.md&#41;)

## 🐈 简介

``中文名：奥夫叉(谐音哈哈🦉)``

```
     ______
 ___ |  ___|_  __
/ _ \| |___\ \/ /
| (_)  | __|\  \     __ _Author : jijue
\___/|_|   /_/\_\__ __ __Version : latest

```

一个应用于web安全领域的漏洞批量扫描框架，可被应用于但不限于如下场景：

> 0Day/1Day全网概念验证(在没有测试环境(各种商业、闭源软件)或懒得搭建测试环境的情况下，直接写POC全网扫描，亲测很爽)

> 企业内网或对外开放资产的安全评估


特点：

* python、跨平台、并发能力强、扫描速度贼快
* 作者持续更新，[POC库](#PocSupport)数量可观
* 功能与时俱进，不断优化打磨
* POC模板结构简洁，提供了专门面对开发者的小工具箱，适合用户自定义拓展POC，能够快速应对不同的安全场景
* 所有POC均以不对目标产生任何负面影响的前提下编写的，旨在发现缺陷的同时不给目标业务造成任何负面影响，能灵活应用于企业与个人不同的安全需求

安全声明：注意，这不是一个黑客工具，仅用于获得企业、单位、组织授权情况下的安全漏洞验证以及安全从业者学习使用，开发者不承担任何相关非法活动的法律责任。

<br>
<br>
<br>
<br>

# ☕️ 使用方法

### 🚀 部署

```
git clone --depth 1 https://github.com/bigblackhat/oFx.git oFx
```

<br>
<br>

### 🍡 用前提醒

使用前提醒，oFx默认每次运行都会有报告和log的输出，分别在output和log文件夹下，出于缓解存储压力的考虑，笔者设定的规则为每次运行时都会删除12小时以前的输出文件，如果该输出文件对使用者有剩余价值，请在运行结束后及时将输出文件移出当前目录或更名为中/英文形式，如果有重要信息被删除，就再扫一遍吧，也快得很

另外，oFx如今可以良好的运行于Mac和Kali系统，相信其他linux也可以正常运行，但由于笔者没有Windows测试环境，因此在Windows上运行oFx可能会发生一些不可预测的错误，请避免在Windows上运行oFx

<br>
<br>

### 🛵 单个url扫描模式

单个url扫描模式的使用场景：
> POC功能性测试

使用方法

```console
➜  oFx git:(main) ✗ python3 ofx.py -s poc/Jboss/Jboss_Unauthorized_access/poc.py -u xxx.xxx.xxx.xxx:xx
```

> 单个目标的漏洞验证详情(
> 返回取决于漏洞本身，目前所有POC都是为了批量验证而设计的，single检测模式尚没有对返回结果做优化，后续会有调整)

![show](img/001.png)

漏洞存在与否见最后一行

<br>
<br>

### ⭕漏洞利用

-e，写webshell、反弹shell等。

目前支持漏洞利用的POC：

```
禅道8.2-9.2.1注入GetShell       写Webshell
CVE-2024-36401 未授权RCE        反弹shell
```

由于笔者有选择困难症，还没想好漏洞利用功能的交互方式，暂时先这么用着吧

### 🚗 批量扫描模式

使用场景：

> 新漏洞爆出来做全网验证

> 刷CNVD之类的漏洞平台的积分或排名

> 有RCE漏洞的POC的话，就可以刷肉鸡(见下方的[POC支持清单](#PocSupport))

使用方法

```sh
➜  oFx git:(main) ✗ python3 ofx.py -s poc/Jboss/Jboss_Unauthorized_access/poc.py -f scan/jboss001.txt --thread 30
```

蓝字代表无漏洞  
![show](img/002.png)

绿字代表存在漏洞  
![show](img/003.png)

青字目标不可达  
![show](img/004.png)

也可以通过``,``分隔同时指定多个poc

```sh
➜  oFx git:(main) ✗ python3 ofx.py -s poc/Jboss/Jboss_Unauthorized_access/poc.py,poc/Jenkins/Unauth_Access_DashBoard/poc.py,poc/Alibaba_Druid/Unauth_Access_DashBoard/poc.py -f scan/jboss001.txt --thread 30
```

最后还可以通过``-s all``指定全量POC进行测试

```sh
➜  oFx git:(main) ✗ python3 ofx.py -s all -f scan/jboss001.txt --thread 50
```

全量POC下测试时常较久，建议食用方式：

* 根据自己电脑性能和带宽给到50个或更多的线程数
* 睡前开始扫描或出门玩儿之前打开oFx进行全量POC测试

偷偷说一句，因为全量POC测试是``依次运行所有POC对目标进行扫描``
，也就是只有一个POC扫完全部目标以后才会调用下一个POC，所以对每一个目标来说可能几十秒甚至几分钟内只有一次恶意请求（时长取决于资产数量和线程数），oFx本身不使用爬虫、fuzzing等技术，不会对被测资产产生任何负载压力，所以即使客户明令禁止使用漏扫，也可以放心大胆的用oFx哦

<br>
<br>

### 🚂 单个目标的全量POC测试

全量POC，顾名思义，遍历所有POC插件依次有序的对目标进行测试，接近于漏扫的概念，而且是一个专注于NDay的漏洞扫描器，因oFx的拓展性，非常适合手中掌握数量庞大的0Day的组织或个人（比如笔者自己），使得他可以将0Day库武器化。

没有被武器化的0Day库，就是一堆无用的文档和漏洞逻辑，没有哪个同行会一边读着上千份文档一边做渗透吧。

开始扫描：

```sh
# Single检测模式不支持全量POC，所以需要先将单个目标保存到文件中，再用批量扫描来扫描，算是曲线救国吧
# 单个或少于十个的目标就没有必要指定线程数了，因为默认线程数是10
echo http://xxx.xxx.com > scan/1.txt
python3 oFx.py -s all -f scan/1.txt
```

<br>
<br>

### 🐾 Fofa api 资产获取

通过fofa提供的api接口获取资产清单并去重

![show](img/005.png)

Fofa Search分成三种模式：  
1）默认模式，即：一条dork语句最大获取1w个资产  
2）自适应模式，即：根据fofa结果，自适应获取当前dork在全球保有量最多的5个国家与其5个行政区划的资产，理论上一条有效dork语句最多可以获得25w资产  
3）中国全境模式，即：根据中国全境行政区划（以Fofa官方设计为准），获取所有资产，理论上一条有效dork语句最多可以获得31w资产

fofa的配置文件位置为：``lib/fofa.ini``

```
备注：
每次运行fofa搜索功能时，oFx都会尝试登陆。
如果失败，会要求用户提供账号和key，此时再登陆成功，oFx会动态的修改配置文件中的user和key，无需打开配置文件调整，下次再使用时也直接生效不必重新输入user和key

在获取资产过程中，如果因报错中断或手动关闭程序，并不影响在此之前获取到的资产保存，可以根据  scan/运行fofa-search后指定的文件名.txt  去找。
```

<br>
<br>

### 🐾 Ceye配置

近期的版本更新中，oFx添加了对Ceye的支持，目前已有一部分POC采用了该平台来辅助漏洞检测，默认下载oFx之后不配置该项并不影响oFx正常的功能使用，只是相关的POC无法按照预期运行而已，有需要可以配置下：

```ini
; 在位于项目根目录下的lib/ceye.ini文件中修改dns和token两项的值
[ceye]
; 从ceye中拿到以下的数据，缺一不可
dns = xxxxxx.ceye.io

token = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

<br>

### 💦 proxypool

近期的版本更新中，oFx提供了对proxypool的支持，proxypool是个http代理池工具，原本定位是用于爬虫使用，防止同一个ip访问次数过多被封ip，该功能也适用于我们漏扫场景，当我们对同一个站点或同一个单位的资产进行一定数量的扫描时，有极大的概率会被封。

proxypool本身有11个免费代理源，程序运行起来以后会定时采集免费代理并验证代理的可用性，同时可以由用户自己添加代理源以提高代理的数量和质量。

项目地址：https://github.com/jhao104/proxy_pool

使用方法：  
proxypool的使用方法可以参考项目文档，oFx与其对接的原理就是访问http://127.0.0.1:5000的api接口，有特殊需要可以自行调整

下面仅描述本地启动proxypool并运行oFx与其对接的方法。

1.下载项目

```shell
git clone git@github.com:jhao104/proxy_pool.git
```

2.安装依赖

```shell
pip install -r requirements.txt
```

3.安装redis，然后修改api与数据库配置

```python
# setting.py 为项目配置文件

# 配置API服务

HOST = "0.0.0.0"  # IP
PORT = 5000  # 监听端口

# 配置数据库

DB_CONN = 'redis://:@127.0.0.1:8888/0'

# 配置 ProxyFetcher

PROXY_FETCHER = [
    "freeProxy01",  # 这里是启用的代理抓取方法名，所有fetch方法位于fetcher/proxyFetcher.py
    "freeProxy02",
    # ....
]

# proxyCheck时代理数量少于POOL_SIZE_MIN触发抓取
POOL_SIZE_MIN = 50  # 这里笔者觉得50会好一点
```

4.启动proxypool

```shell
# 如果已经具备运行条件, 可用通过proxyPool.py启动。
# 程序分为: schedule 调度程序 和 server Api服务

# 启动调度程序
python proxyPool.py schedule

# 启动webApi服务
python proxyPool.py server

```

5.oFx对接proxypool

直接在后面跟上``--proxypool``参数即可。

```shell
python3 ofx.py -s all -f xxx.txt --proxypool --thread 50
```

此时oFx就会通过proxypool的api接口获取一个代理，然后判断：

* 如果是国外的代理，就会抛弃并删除该代理，然后重新获取。
* 如果是国内代理，就会尝试进行访问测试，如果失败，同样抛弃并删除，但如果测试成功，则会引用该代理。

# 🐇 POC支持清单<div id="PocSupport"></div>

<!--
<br>

<details>
<summary>支持的漏洞列表 [点击展开] </summary>  
-->

| 应用                        | 漏洞名称                                                            | POC路径                                                                      |
|---------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------|
| 360                       | 360天擎数据库未授权访问                                                   | ``poc/360/TianQing_Unauth_Acceess/poc.py``                                 |
|                           | 360天擎前台SQL注入                                                    | ``poc/360/TianQing_SQLi/poc.py``                                           |
| ACME                      | mini_httpd任意文件读取漏洞(CVE-2018-18778)                              | ``poc/ACME/File_Read_mini_httpd_CVE_2018_18778/poc.py``                    |
| AJ_Report                 | AJ-Report 身份验证绕过和远程代码执行 (CNVD-2024-15077)                       | ``poc/AJ_Report/RCE_CNVD_2024_15077/poc.py``                               |
| Alibaba_Alist             | AList云盘 未授权访问                                                   | ``poc/Alibaba_Alist/UnAuth_Access/poc.py``                                 |
| Alibaba_Canal             | Alibaba Canal 默认弱口令漏洞                                           | ``poc/Alibaba_Canal/Weak_Pass/poc.py``                                     |
|                           | Alibaba Canal 信息泄露                                              | ``poc/Alibaba_Canal/Info_Disclosure/poc.py``                               |
| Alibaba_Druid             | Druid未授权访问                                                      | ``poc/Alibaba_Druid/Unauth_Access/poc.py``                                 |
| Alibaba_Fastjson          | Fastjson 反序列化远程代码执行漏洞（CVE-2017-18349）                           | ``poc/Alibaba_FastJson/RCE_CVE_2017_18349/poc.py``                         |
| Alibaba_Nacos             | Nacos未授权访问                                                      | ``poc/Alibaba_Nacos/Unauth_Access/poc.py``                                 |
| Apache ActiveMQ           | Apache ActiveMQ 远程代码执行漏洞(CVE-2016-3088)                         | ``poc/Apache_ActiveMQ/RCE_FileServer_CVE_2016_3088/poc.py``                |
|                           | Apache ActiveMQ 弱口令 ➕ CVE-2015-5254                             | ``poc/Apache_ActiveMQ/WeakPass/poc.py``                                    |
|                           | ActiveMQ物理路径泄漏漏洞                                                | ``poc/Apache_ActiveMQ/Physical_Path_Disclosure/poc.py``                    |
| Apache_ApiSix             | Apache APISIX Dashboard 身份验证绕过漏洞（CVE-2021-45232）                | ``poc/Apache_ApiSix/DashBoard_Auth_Bypass_CVE_2021_45232/poc.py``          |
|                           | Apache APISIX batch-requests 未授权RCE(CVE-2022-24112)             | ``poc/Apache_ApiSix/RCE_CVE_2022_24112/poc.py``                            |
|                           | Apache APISIX 默认密钥漏洞（CVE-2020-13945）                            | ``poc/Apache_ApiSix/Default_Key_CVE_2020_13945/poc.py``                    |
| Apache CouchDB            | Apache Couchdb 远程权限提升(CVE-2017-12635)                           | ``poc/Apache_CouchDB/Priv_Escalation_CVE-2017_12635/poc.py``               |
| Apache Druid              | Apache Druid任意文件读取复现(CVE-2021-36749)                            | ``poc/Apache_Druid/File_Read_CVE_2021_36749/poc.py``                       |
|                           | Apache Druid 远程代码执行漏洞 (CVE-2021-25646)                          | ``poc/Apache_Druid/RCE_CVE_2021_25646/poc.py``                             |
| Apache Flink              | Apache Flink目录穿透(CVE-2020-17519)                                | ``poc/Apache_Flink/Dir_Traversal_CVE_2020_17519/poc.py``                   |
|                           | Apache Flink <= 1.9.1远程代码执行 CVE-2020-17518                      | ``poc/Apache_Flink/RCE_CVE_2020_17518/poc.py``                             |
| Apache_Hadoop             | Apache Hadoop反序列化漏洞(CVE-2021-25642)                             | ``poc/Apache_Hadoop/RCE_CVE_2021_25642/poc.py``                            |
| Apache Kylin              | Apache Kylin 未授权配置泄露 CVE-2020-13937                             | ``poc/Apache_Kylin/Conf_Info_Disclosure_CVE_2020_13937/poc.py``            |
| Apache Mod_jk             | Apache Mod_jk 访问控制权限绕过(CVE-2018-11759)                          | ``poc/Apache_Mod_jk/ACbypass_CVE_2018_11759/poc.py``                       |
| Apache Shiro              | Shiro 1.2.4 反序列化漏洞                                              | ``poc/Apache_Shiro/RCE_CVE_2016_4437/poc.py``                              |
|                           | Shiro指纹识别                                                       | ``poc/Apache_Shiro/FingerPrint/poc.py``                                    |
| Apache Solr               | Apache Solr Velocity 注入远程命令执行漏洞 (CVE-2019-17558)                | ``poc/Apache_Solr/CVE_2019_17558/poc.py``                                  |
|                           | Apache Solr 任意文件读取漏洞                                            | ``poc/Apache_Solr/File_Read/poc.py``                                       |
|                           | Apache Solr 远程命令执行 Log4j                                        | ``poc/Apache_Solr/RCE_Log4j_CVE_2021_44228/poc.py``                        |
| Apache ZooKeeper          | ZooKeeper未授权访问                                                  | ``poc/Apache_ZooKeeper/Unauth_Access/poc.py``                              |
| 碧海威 L7                    | 碧海威 L7 弱口令漏洞                                                    | ``poc/Bithighway_碧海威/Weak_Pass_L7/poc.py``                                 |
| BSPHP                     | BSPHP 未授权访问 信息泄露漏洞                                              | ``poc/BSPHP/Info_Disclosure/poc.py``                                       |
| C-Lodop                   | C-Lodop 云打印机系统平台任意文件读取漏洞                                        | ``poc/C_Lodop/File_Read/poc.py``                                           |
| 中国电信                      | 电信天翼网关F460 web_shell_cmd.gch 远程命令执行漏洞                           | ``poc/China_TeleCOM_中国电信/RCE_F460_GateWay/poc.py``                         |
|                           | 大唐电信AC集中管理平台默认口令                                                | ``poc/China_TeleCOM_中国电信/Weak_Pass_DaTang_AC_Manager/poc.py``              |
| 中国移动                      | 中国移动 禹路由 ExportSettings.sh 敏感信息泄露漏洞                             | ``poc/China_Mobile_中国移动/Info_Disclosure_Yu_routing_ExportSettings/poc.py`` |
| CmsEasy                   | CmsEasy crossall_act.php SQL注入漏洞                                | ``poc/CmsEasy/SQLi_CrossAllAct_php/poc.py``                                |
| common(通用)                | git信息泄露                                                         | ``poc/common/Git_Info_Disclosure/poc.py``                                  |
|                           | svn信息泄露                                                         | ``poc/common/Svn_Info_Disclosure/poc.py``                                  |
|                           | URL存活检测                                                         | ``poc/common/Url_Alive/poc.py``                                            |
|                           | Apache列目录                                                       | ``poc/common/Apache_Dir_List/poc.py``                                      |
|                           | 备份泄露/源码泄露                                                       | ``poc/common/CodeDisclosure/poc.py``                                       |
| Confluence                | Confluence Server Webwork OGNL注入 PreAuth-RCE(CVE-2021-26084)    | ``poc/Confluence/OGNL_Injection_CVE_2021_26084/poc.py``                    |
|                           | Atlassian Confluence OGNL表达式注入漏洞（CVE-2022-26134）                | ``poc/Confluence/OGNL_Injection_CVE_2022_26134/poc.py``                    |
|                           | Confluence 路径穿越与命令执行（CVE-2019-3396）                             | ``poc/Confluence/Path_Travel_CVE_2019_3396/poc.py``                        |
| Coremail                  | Coremail 配置信息泄露漏洞                                               | ``poc/Coremail/Conf_Info_Disclosure/poc.py``                               |
| 赤兔CMS                     | 赤兔CMS banner识别插件                                                | ``poc/CtCMS_赤兔CMS/Get_Banner/poc.py``                                      |
| D-Link                    | D-Link ShareCenter DNS-320 system_mgr.cgi 远程命令执行漏洞              | ``poc/D_Link/RCE_ShareCenter_system_mgr_cgi/poc.py``                       |
|                           | D-Link Dir-645 getcfg.php 账号密码泄露漏洞(CVE-2019-17506)              | ``poc/D_Link/UPInfo_Disclosure_getcfg_php/poc.py``                         |
|                           | D-Link AC管理系统默认账号密码                                             | ``poc/D_Link/Weak_Pass_AC_Manager/poc.py``                                 |
| Dapr                      | Dapr Dashboard未授权访问(CVE-2022-38817)                             | ``poc/Dapr/Unauth_Access_DashBoard/poc.py``                                |
| DBAppSecurity 安恒信息        | 明御WAF登陆绕过                                                       | ``poc/dbappsecurity_安恒信息/Login_ByPass_MingYu_WAF/poc.py``                  |
| 织梦CMS                     | 织梦CMS radminpass.php文件暴露                                        | ``poc/DedeCMS_织梦/RadminPass/poc.py``                                       |
|                           | DedeCMS 短文件名信息泄露                                                | ``poc/DedeCMS_织梦/Info_Disclosure_IIS_Short_Filename/poc.py``               |
| DocCMS                    | DocCMS keyword SQL注入漏洞                                          | ``poc/DocCMS/SQLi_keyword/poc.py``                                         |
| Docker                    | Docker Remote API未授权访问                                          | ``poc/Docker/Unauth_Remote_Api/poc.py``                                    |
| DrayTek                   | DrayTek企业网络设备 远程命令执行(CVE-2020-8515)                             | ``poc/DrayTek/RCE_CVE_2020_8515/poc.py``                                   |
| Drupal!                   | Drupal!远程代码执行(CVE-2018-7600)                                    | ``poc/Drupal!/RCE_CVE_2018_7600/poc.py``                                   |
| DVR                       | DVR登录绕过漏洞(CVE-2018-9995)                                        | ``poc/DVR/Login_Bypass_CVE_2018_9995/poc.py``                              |
| ECShop                    | ECShop 4.1.0前台 delete_cart_goods.php SQL注入(CNVD-2020-58823)     | ``poc/ECShop/SQLi_delete_cart_goods/poc.py``                               |
|                           | ECShop 2.x/3.x SQL 注入/远程代码执行漏洞                                  | ``poc/ECShop/RCE_2dotX_OR_3dotX/poc.py``                                   |
| ElasticSearch             | ElasticSearch 未授权访问                                             | ``poc/Elasticsearch/Unauth_Access/poc.py``                                 |
|                           | ElasticSearch 命令执行漏洞（CVE-2014-3120）                             | ``poc/Elasticsearch/Cmd_Exec_MVEL_CVE-2014-3120/poc.py``                   |
|                           | ElasticSearch Groovy 沙盒绕过 && 代码执行漏洞（CVE-2015-1427）              | ``poc/Elasticsearch/Code_Exec_Groovy_CVE-2015-1427/poc.py``                |
|                           | ElasticSearch 目录穿越漏洞（CVE-2015-5531）                             | ``poc/Elasticsearch/Dir_Traversal_CVE-2015-5531/poc.py``                   |
|                           | Elasticsearch写任意文件漏洞（WooYun-2015-110216）                        | ``poc/Elasticsearch/File_Create_WooYun-2015-110216/poc.py``                |
| EnjoySCM_捷诚               | EnjoySCM UploadFile任意文件上传漏洞                                     | ``poc/EnjoySCM_捷诚/FileUpload_UploadFile/poc.py``                           |
| Eyou 亿邮电子邮件系统             | 亿邮电子邮件系统 远程命令执行                                                 | ``poc/Eyou_亿邮/RCE_moni_detail/poc.py``                                     |
| F5                        | F5 BIG-IP任意文件读取(CVE-2020-5902)                                  | ``poc/F5_BIG_IP/File_Read_CVE_2020_5902/poc.py``                           |
|                           | CVE-2021-22986 RCE                                              | ``CVE-2021-22986 RCE``                                                     |
| FineReport_帆软             | 帆软报表 V8 get_geo_json 任意文件读取漏洞                                   | ``poc/FineReport_帆软/FileRead8/poc.py``                                     |
| 菲力尔                       | FLIR-AX8 download.php 任意文件下载                                    | ``poc/FLIR_菲力尔/Download_File_AX8/poc.py``                                  |
| Fortinet_美国飞塔             | 飞塔(Fortinet)防火墙身份认证绕过漏洞(CVE-2022-40684)                         | ``poc/Fortinet_美国飞塔/Authentication_Bypass_FortiOS_CVE_2022_40648/poc.py``  |
|                           | Fortigate SSL VPN fgt_lang 任意文件读取                               | ``poc/Fortinet_美国飞塔/FileRead_SSL_VPN/poc.py``                              |
| FTP                       | FTP匿名访问                                                         | ``poc/FTP/Anonymous_Access/poc.py``                                        |
| GeoServer                 | CVE-2024-36401 未授权RCE                                           | `poc/GeoServer/RCE_CVE_2024_36401/poc.py`                                  |
| Grafana                   | Grafana plugins 任意文件读取漏洞(CVE-2021-43798)                        | ``poc/Grafana/File_Read_plugins/poc.py``                                   |
| H2 数据库                    | H2 数据库 Web控制台未授权访问                                              | ``poc/H2_DataBase/UnAuth_Access/poc.py``                                   |
| H3C SecPath 下一代防火墙        | H3C SecPath 下一代防火墙 任意文件下载漏洞                                     | ``poc/H3C/File_Download_SecPath_WAF/poc.py``                               |
| 海康威视                      | HIKVISION 视频编码设备接入网关 任意文件下载                                     | ``poc/HIKVISION/File_Down_Gateway_downFile_php/poc.py``                    |
|                           | HIKVISION 流媒体管理服务器弱口令                                           | ``poc/HIKVISION/Weak_Pass_Stream_Media_Manager/poc.py``                    |
|                           | HIKVISION 流媒体管理服务器任意文件读取                                        | ``poc/HIKVISION/File_Read_Stream_Media_Manager/poc.py``                    |
|                           | 海康威视 IP摄像头未授权访问（CVE-2017-7921）                                  | ``poc/HIKVISION/UnAuth_Access_CVE_2017_7921/poc.py``                       |
|                           | HIKVISION 综合安防管理平台 applyCT Fastjson远程命令执行漏洞                     | ``poc/HIKVISION/RCE_applyCT_FastJson/poc.py``                              |
|                           | 海康威视 IP Camera 远程命令执行漏洞（CVE-2021-36260）                         | ``poc/HIKVISION/RCE_IP_Camera_CVE_2021_36260/poc.py``                      |
| 宏电                        | 宏电 H8922 后台任意文件读取漏洞                                             | ``poc/Hongdian_宏电/Backstage_File_Read_CVE_2021_28152/poc.py``              |
| 好视通                       | 好视通视频会议平台 任意文件下载                                                | ``poc/HST_好视通/File_Download/poc.py``                                       |
| 华为                        | Huawei HG659 lib 任意文件读取漏洞                                       | ``poc/Huawei/File_Read_HG659_lib/poc.py``                                  |
|                           | 华为路由器敏感信息泄露 DG8045 Router 1.0                                   | ``poc/Huawei/Info_Disclosure_DG8045/poc.py``                               |
| 汇文                        | 汇文OPAC敏感信息泄露                                                    | ``poc/HuiWen_汇文/Info_Disclosure/poc.py``                                   |
|                           | 汇文OPAC弱口令                                                       | ``poc/HuiWen_汇文/Weak_Pass/poc.py``                                         |
| 蜂网互联                      | 蜂网互联 企业级路由器v4.31 密码泄露漏洞                                         | ``poc/IFW8_蜂网互联/UPInfo_DisClosure_CVE_2019_16313/poc.py``                  |
| InfluxDB                  | InfluxDB指纹识别                                                    | ``poc/InfluxDB/FingerPrint/poc.py``                                        |
|                           | InfluxDB 未授权访问                                                  | ``poc/InfluxDB/UnAuth_Access/poc.py``                                      |
| Intelbras                 | Intelbras Wireless 未授权与密码泄露                                     | ``poc/Intelbras/UPInfo_Disclosure_CVE_2021_3017/poc.py``                   |
| Jboss                     | Jboss未授权访问                                                      | ``poc/Jboss/Unauth_Access/poc.py``                                         |
| Jellyfin                  | Jellyfin任意文件读取                                                  | ``poc/jellyfin/File_Read_CVE_2021_21402/poc.py``                           |
|                           | Jellyfin RemoteImageController.cs SSRF漏洞(CVE-2021-29490)        | ``poc/jellyfin/SSRF_CVE_2021_29490/poc.py``                                |
| Jenkins                   | Jenkins未授权访问                                                    | ``poc/Jenkins/Unauth_Access/poc.py``                                       |
| Jetty                     | Jetty WEB-INF文件读取漏洞(CVE-2021-34429)                             | ``poc/Jetty/File_Read_CVE_2021_34429/poc.py``                              |
|                           | Jetty指纹识别                                                       | ``poc/Jetty/FingerPrint/poc.py``                                           |
|                           | Jetty WEB-INF 敏感信息泄露漏洞（CVE-2021-28164）                          | ``poc/Jetty/Info_Disclosure_CVE_2021_28164/poc.py``                        |
|                           | Jetty Utility Servlets ConcatServlet 双解码信息泄露漏洞 (CVE-2021-28169) | ``poc/Jetty/Info_Disclosure_CVE_2021_28169/poc.py``                        |
| JimuReport                | JimuReport FreeMarker 服务端模板注入命令执行（CVE-2023-4450）                | ``poc/JimuReport/RCE_CVE_2023_4450/poc.py``                                |
| 金和OA                      | 金和OA C6 download.jsp 任意文件读取漏洞                                   | ``poc/Jinher_金和OA/File_Read_download_jsp/poc.py``                          |
| KEDACOM 数字系统接入网关          | KEDACOM 数字系统接入网关 任意文件读取漏洞                                       | ``poc/KEDACOM_数字系统接入网关/File_Read/poc.py``                                  |
| 金蝶OA                      | 金蝶协同办公系统 fileDownload.do 任意文件下载漏洞                               | ``poc/Kingdee_金蝶/File_Down_fileDownload_do/poc.py``                        |
|                           | 金蝶OA server_file 目录遍历漏洞                                         | ``poc/Kingdee_金蝶/Dir_List_server_file/poc.py``                             |
| Kyan网络监控设备                | Kyan网络监控设备信息泄露                                                  | ``poc/Kyan/Info_Disclosure/poc.py``                                        |
| 蓝凌OA                      | 蓝凌OA custom.jsp 前台任意文件读取漏洞                                      | ``poc/Landray_蓝凌OA/File_Read_CNVD_2021_28277/poc.py``                      |
|                           | 蓝凌OA treexml.tmpl脚本 远程命令执行                                      | ``poc/Landray_蓝凌OA/RCE_TreeXmlTMPL_Script/poc.py``                         |
|                           | 蓝凌OA sysFormulaSimulateByJS 未授权命令执行漏洞                           | ``poc/Landray_蓝凌OA/RCE_sysFormulaSimulateByJS/poc.py``                     |
| Laravel Framework         | Laravel .env 配置文件泄露                                             | ``poc/Laravel_Framework/Conf_Info_Disclosure_dot_env/poc.py``              |
| 朗驰欣创                      | 朗驰欣创视频监控系统 FTP账号密码泄露                                            | ``poc/LinkSeek_朗驰欣创/FTP_Account_Info_Disclosure/poc.py``                   |
| 利谱第二代防火墙                  | 利谱第二代防火墙存在信息泄露漏洞                                                | ``poc/LiPu_利谱第二代防火墙/Info_Disclosure/poc.py``                               |
| 佑友                        | 佑友防火墙 弱口令                                                       | ``poc/MailGard_佑友/Weak_Pass_FireWall/poc.py``                              |
|                           | 佑友防火墙 后台命令执行漏洞                                                  | ``poc/MailGard_佑友/RCE_ping_FireWall/poc.py``                               |
| 迈普 ISG1000安全网关            | 迈普 ISG1000安全网关 任意文件下载漏洞                                         | ``poc/MaiPu_迈普/File_Download_webui/poc.py``                                |
| MemCache                  | MemCache未授权访问                                                   | ``poc/MemCache/UnAuth_Access/poc.py``                                      |
| MessageSolution企业邮件归档管理系统 | MessageSolution企业邮件归档管理系统 EEA 信息泄露                              | ``poc/MessageSolution/Info_Disclosure/poc.py``                             |
| MetaBase                  | MetaBase任意文件读取漏洞 CVE-2021-41277                                 | ``poc/Metabase/File_Read_CVE_2021_41277/poc.py``                           |
| MicroSoft                 | Windows HTTP协议栈远程代码执行漏洞(CVE-2022-21907)                         | poc/MicroSoft/RCE_CVE_2022_21907/poc.py                                    |
| MongoDB                   | MongoDB未授权访问                                                    | ``poc/MongoDB/Unauth_Access/poc.py``                                       |
| Mysql                     | Mysql弱口令                                                        | ``poc/Mysql/WeakPass/poc.py``                                              |
| 蓝海卓越                      | 蓝海卓越计费管理系统 任意文件读取                                               | ``poc/NatShell_蓝海卓越/File_Read/poc.py``                                     |
|                           | 蓝海卓越计费管理系统 认证hash泄露                                             | ``poc/NatShell_蓝海卓越/HashInfo_DisClosure/poc.py``                           |
| 中科网威                      | 中科网威 下一代防火墙控制系统 账号密码泄露漏洞                                        | ``poc/NetPower_中科网威/UPInfo_DisClosure_Firewall/poc.py``                    |
| Node.js                   | Node.js目录穿越漏洞                                                   | ``poc/Node.js/Dir_Traversal_CVE_2017_14849/poc.py``                        |
|                           | Node.js命令注入漏洞（CVE-2021-21315）                                   | ``poc/Node.js/Cmd_inj_CVE_2021_21315/poc.py``                              |
| 绿盟                        | 绿盟下一代防火墙 resourse.php 任意文件上传漏洞                                  | ``poc/nsfocus_绿盟/File_upload_NF/poc.py``                                   |
| 新软科技                      | 极通EWEBS应用虚拟化系统任意文件读取                                            | ``poc/NSoft_新软/FileRead_EWEBS/poc.py``                                     |
| OKI                       | OKI MC573未授权访问                                                  | ``poc/OKI/UnAuth_MC573/poc.py``                                            |
| 梨子项目管理系统                  | 梨子项目管理系统 信息泄露漏洞                                                 | ``poc/PearProject_梨子项目管理系统/Conf_Info_Disclosure_env/poc.py``               |
| PHP                       | php v8.1开发版后门检测                                                 | ``poc/php/Backdoor_v8dev/poc.py``                                          |
| PHPStudy                  | PHPStudy 后门检测                                                   | ``poc/PHPStudy/Back_Door/poc.py``                                          |
| PHPUnit                   | PHPUnit eval-stdin.php 远程命令执行漏洞                                 | ``poc/PHPUnit/RCE_eval_stdin/poc.py``                                      |
| QZSec_齐治                  | 齐治堡垒机 任意用户登录漏洞                                                  | ``poc/QZSec_齐治/AnyUser_Login_Fortress_Machine/poc.py``                     |
| Redis                     | Redis未授权访问                                                      | ``poc/Redis/Unauth_Access/poc.py``                                         |
|                           | Redis Lua 沙箱逃逸和远程代码执行 (CVE-2022-0543)                           | ``poc/Redis/RCE_Lua_Sandbox_Escape_CVE_2022_0543/poc.py``                  |
| RLM                       | Reprise License Manager 14.2 信息泄露（CVE-2022-28365）               | ``poc/RLM/Info_Disclosure_CVE_2022_28365/poc.py``                          |
| 锐捷                        | 锐捷EG网关 userAuth.php存在任意文件读取漏洞                                   | ``poc/Ruijie_锐捷/File_Read_EG_userAuth/poc.py``                             |
|                           | 锐捷NBR 1300G 路由器 越权CLI命令执行漏洞                                     | ``poc/Ruijie_锐捷/RCE_NBR_1300G/poc.py``                                     |
|                           | 锐捷NBR路由器 EWEB网管系统 远程命令执行漏洞                                      | ``poc/Ruijie_锐捷/RCE_EWEB_Manager_CNVD_2021_09650/poc.py``                  |
|                           | 锐捷RG-UAC/RG-ISG统一上网行为管理审计系统存在账号密码信息泄露                           | ``poc/Ruijie_锐捷/UPInfo_DisClosure_RG_UAC_CNVD_2021_14536/poc.py``          |
|                           | 锐捷Smartweb管理系统 默认账户➕命令执行漏洞                                      | ``poc/Ruijie_锐捷/RCE_SmartWeb_WEB_VMS/poc.py``                              |
|                           | 锐捷云课堂主机 目录遍历漏洞                                                  | ``poc/Ruijie_锐捷/Dir_List_Cloud_ClassRoom/poc.py``                          |
| 若依后台管理系统                  | 若依后台管理系统 弱口令                                                    | ``poc/RuoYi_若依/Weak_Pass/poc.py``                                          |
| Samsung                   | 三星路由器本地文件包含                                                     | ``poc/Samsung/Lfi_Samsung_Wlan_AP/poc.py``                                 |
|                           | 三星 WLAN AP WEA453e路由器 远程命令执行漏洞                                  | ``poc/Samsung/RCE_Samsung_WLANAP_WEA453e/poc.py``                          |
| Sangfor 深信服               | 深信服EDR终端检测响应平台RCE漏洞(CNVD-2020-46552)                            | ``poc/SANGFOR_深信服/RCE_2020_EDR/poc.py``                                    |
| Sapido                    | Sapido BRC70n路由器远程代码执行漏洞                                        | ``poc/Sapido/RCE_BRC70n_Router/poc.py``                                    |
| 致远OA                      | 致远OA webmail.do 任意文件下载 (CNVD-2020-62422)                        | ``poc/SeeYon_致远/File_Download/poc.py``                                     |
|                           | 致远OA ajax.do 任意文件上传                                             | ``poc/SeeYon_致远/File_Upload_ajax_do/poc.py``                               |
|                           | 致远OA getSessionList.jsp Session泄漏漏洞                             | ``poc/SeeYon_致远/Session_Disclosure_getSessionList/poc.py``                 |
|                           | 致远OA A6 test.jsp SQL注入漏洞                                        | ``poc/SeeYon_致远/SQLi_test_jsp/poc.py``                                     |
|                           | Sophos Mobile OmaDsServlet XML实体注入(CVE-2022-3980)               | ``poc/Sophos/XXE_CVE_2022_3980/poc.py``                                    |
| 狮子鱼CMS                    | 狮子鱼CMS ApiController.class.php SQL注入漏洞                          | ``poc/ShiZiYu_狮子鱼/Sqli_ApiController/poc.py``                              |
|                           | 狮子鱼CMS ApigoodsController.class.php SQL注入漏洞                     | ``poc/ShiZiYu_狮子鱼/Sqli_ApigoodsController/poc.py``                         |
| ShopXO                    | ShopXO download 任意文件读取漏洞(CNVD-2021-15822)                       | ``poc/ShopXO/FileRead_CNVD_2021_15822/poc.py``                             |
| SonarQube                 | SonarQube api 信息泄露漏洞                                            | ``poc/SonarQube/Info_Disclosure_CVE_2020_27986/poc.py``                    |
| SonicWall SSL-VPN         | SonicWall SSL-VPN 远程命令执行漏洞                                      | ``poc/SonicWall_SSL_VPN/RCE_jarrewrite/poc.py``                            |
| TamronOS IPTV系统           | TamronOS IPTV系统 后台配置敏感信息                                        | ``poc/TamronOS_IPTV/Info_Disclosure/poc.py``                               |
|                           | TamronOS IPTV系统存在前台命令执行漏洞                                       | ``poc/TamronOS_IPTV/RCE_api_ping/poc.py``                                  |
|                           | TamronOS IPTV系统 submit 任意用户创建漏洞                                 | ``poc/TamronOS_IPTV/User_Add_Submit/poc.py``                               |
| TCC_斗象                    | 斗象资产灯塔系统(ARL) 弱口令检测                                             | ``poc/TCC_斗象/Weak_Pass_ARL/poc.py``                                        |
|                           | Omnia MPX Node 不安全的直接对象引用(CVE-2022-43326)                       | ``poc/Telos_Alliance/IDOR_Omnia_MPX_Node_CVE_2022_43326/poc.py``           |
| ThinkPHP                  | ThinkPHP5 5.0.22/5.1.29 远程代码执行漏洞                                | ``poc/Thinkphp/RCE_5022_5129``                                             |
|                           | ThinkPHP5 5.0.23 远程代码执行漏洞                                       | ``poc/Thinkphp/RCE_5023/poc.py``                                           |
| 通达OA                      | 通达OA 计算机名探测插件                                                   | ``poc/Tongda_通达OA/Computer_Name_Plugin/poc.py``                            |
|                           | 通达OA 版本探测插件                                                     | ``poc/Tongda_通达OA/Version_Info_Plugin/poc.py``                             |
|                           | 通达OA2017 前台任意用户登录漏洞                                             | ``poc/Tongda_通达OA/AnyUser_Login_Version2017/poc.py``                       |
|                           | 通达OA v2017 video_file.php 任意文件下载漏洞                              | ``poc/Tongda_通达OA/FileRead_video_file/poc.py``                             |
|                           | 通达OA retrieve_pwd 任意用户登陆漏洞                                      | ``poc/Tongda_通达OA/AnyUser_Login_retrieve_pwd/poc.py``                      |
|                           | 通达OA < v11.7 auth_mobi.php 在线用户登录漏洞                             | ``poc/Tongda_通达OA/InfoDisclosure_auth_mobi/poc.py``                        |
|                           | 通达OA v11.9 getdata 任意命令执行漏洞                                     | ``poc/Tongda_通达OA/RCE_getdata/poc.py``                                     |
| 天融信                       | 天融信-上网行为管理系统 static_convert.php 命令注入漏洞                          | ``poc/TopSec_天融信/Common_Injection_static_convert/poc.py``                  |
| 同为股份                      | TVT数码科技 NVMS-1000 路径遍历漏洞                                        | ``poc/TVT_同为股份/Dir_Traversal_NVMS_1000/poc.py``                            |
| 艾泰科技                      | 艾泰网络管理系统弱口令                                                     | ``poc/UTT_艾泰科技/WeakPass_Net_Manager_System/poc.py``                        |
| 启明星辰                      | 天玥运维网关/网御网络审计 Sql注入漏洞                                           | ``poc/Venustech_启明星辰/SQLi_Reportguide/poc.py``                             |
| VMware                    | Vmware vCenter 任意文件读取                                           | ``poc/VMware/File_read_vCenter/poc.py``                                    |
|                           | VMware vRealize Operations Manager SSRF漏洞 CVE-2021-21975        | ``poc/VMware/SSRF_vRealize_CVE_2021_21975/poc.py``                         |
|                           | VMware Workspace ONE Access Freemarker 服务器端模板注入(CVE-2022-22954) | ``poc/VMware/CVE_2022_22954/poc.py``                                       |
|                           | VMware NSX Manager XStream 远程代码执行漏洞（CVE-2021-39144）             | ``poc/VMware/RCE_NSX_Manager_XStream_CVE_2021_39144/poc.py``               |
| VoIPmonitor               | VoIPmonitor 未授权远程代码执行(CVE-2021-30461)                           | ``poc/VoIPmonitor/RCE_CVE_2021_30461/poc.py``                              |
| WayOS 维盟                  | 维盟AC集中管理平台弱口令                                                   | ``poc/WayOS_维盟/WeakPass_AC_Manager/poc.py``                                |
| 泛微 OA                     | 泛微云桥 e-Bridge 任意文件读取漏洞                                          | ``poc/Weaver_泛微OA/File_Read_E_Bridge/poc.py``                              |
|                           | 泛微OA E-Office V9文件上传漏洞(CNVD-2021-49104)                         | ``poc/Weaver_泛微OA/File_Upload_E_Office_V9_CNVD_2021_49104/poc.py``         |
|                           | 泛微 e-cology OA 数据库配置信息泄露漏洞                                      | ``poc/Weaver_泛微OA/Config_Info_Disclosure_DBconfigReader/poc.py``           |
|                           | 泛微 OA 8 前台SQL注入                                                 | ``poc/Weaver_泛微OA/Sql_inj_E_cology_V8/poc.py``                             |
|                           | 泛微OA 日志泄露                                                       | ``poc/Weaver_泛微OA/Log_Disclosure/poc.py``                                  |
|                           | 泛微OA Beanshell 远程代码执行漏洞                                         | ``poc/Weaver_泛微OA/RCE_Beanshell/poc.py``                                   |
|                           | 泛微 E-cology WorkflowCenterTreeData.jsp文件 前台SQL注入漏洞              | ``poc/Weaver_泛微OA/Sql_Inj_E_cology_WorkflowCenterTreeData/poc.py``         |
|                           | 泛微V9 前台文件上传漏洞                                                   | ``poc/Weaver_泛微OA/File_Upload_V9_uploadOperation/poc.py``                  |
|                           | 泛微 E-cology V9信息泄露                                              | ``poc/Weaver_泛微OA/Config_Info_Disclosure_E_Cology_V9/poc.py``              |
|                           | 泛微 E-Office存在前台文件上传漏洞                                           | ``poc/Weaver_泛微OA/File_Upload_E_Office_ajax/poc.py``                       |
|                           | 泛微 E-office V9.5 SQL注入漏洞                                        | ``poc/Weaver_泛微OA/SQLi_E_Office_v9dot5/poc.py``                            |
|                           | 泛微OA E-Bridge saveYZJFile 任意文件读取漏洞                              | ``poc/Weaver_泛微OA/FileRead_saveYZJFile/poc.py``                            |
|                           | 泛微OA E-Office officeserver.php 任意文件读取漏洞                         | ``poc/Weaver_泛微OA/File_Read_E_Office_officeserver/poc.py``                 |
|                           | 泛微OA weaver.common.Ctrl 任意文件上传漏洞                                | ``poc/Weaver_泛微OA/File_Upload_weaver_common_ctrl/poc.py``                  |
|                           | 泛微e-cology登陆绕过                                                  | ``poc/Weaver_泛微OA/Login_Pass/poc.py``                                      |
|                           | 泛微云桥(e-bridge) addTaste存在sql注入漏洞                                | ``poc/Weaver_泛微OA/SQLi_E_Bridge_addTaste/poc.py``                          |
|                           | 泛微OA E-Office mysql_config.ini 数据库信息泄漏                          | ``poc/Weaver_泛微OA/InfoDisclosure_mysql_config_ini/poc.py``                 |
|                           | 泛微OA E-Weaver SignatureDownLoad 任意文件读取漏洞                        | ``poc/Weaver_泛微OA/FileRead_SignatureDownLoad/poc.py``                      |
| Weblogic                  | CVE-2016-0638                                                   | ``poc/Weblogic/CVE_2016_0638/poc.py``                                      |
|                           | Weblogic < 10.3.6 'wls-wsat' XMLDecoder 反序列化漏洞（CVE-2017-10271）  | ``poc/Weblogic/CVE_2017_10271/poc.py``                                     |
|                           | RCE_CVE-2018-3191                                               | ``poc/Weblogic/RCE_CVE_2018_3191/poc.py``                                  |
|                           | Weblogic SSRF (CVE-2014-4210)                                   | ``poc/Weblogic/SSRF_CVE_2014_4210/poc.py``                                 |
|                           | Weblogic 管理控制台未授权远程命令执行漏洞（CVE-2020-14882，CVE-2020-14883）        | ``poc/Weblogic/UnAuth_RCE_CVE_2020_14882/poc.py``                          |
|                           | Weblogic XMLDecoder反序列化漏洞（CVE-2017-3506）                        | ``poc/Weblogic/XMLDecoder_CVE_2017_3506/poc.py``                           |
| whir_万户软件                 | 万户ezoffice 前台sql注入                                              | ``poc/whir_万户软件/SQLi_EZOffice/poc.py``                                     |
|                           | 万户OA DownloadServlet 任意文件读取漏洞                                   | ``poc/whir_万户软件/FileDownload_DownloadServlet/poc.py``                      |
|                           | 万户OA downloadhttp.jsp 任意文件下载漏洞                                  | ``poc/whir_万户软件/FileDownload_downloadhttp_jsp/poc.py``                     |
|                           | 万户OA download_old.jsp 任意文件下载漏洞                                  | ``poc/whir_万户软件/FileDownload_download_old_jsp/poc.py``                     |
| 一米OA                      | 一米OA getfile.jsp 任意文件读取漏洞                                       | ``poc/YimiOA/ReadFile_getfile/poc.py``                                     |
| 用友NC                      | 用友NC6.5 BeanShell RCE                                           | ``poc/Yonyou_用友NC/RCE_BeanShell_CNVD_2021_30167/poc.py``                   |
|                           | 用友ERP-NC 目录遍历漏洞                                                 | ``poc/Yonyou_用友NC/Dir_List_ERP/poc.py``                                    |
|                           | 用友GRP-U8行政事业财务管理软件 SQL注入 CNNVD-201610-923                       | ``poc/Yonyou_用友NC/Sqli_CNNVD_201610_923/poc.py``                           |
|                           | 用友 时空KSOA 前台文件上传漏洞                                              | ``poc/Yonyou_用友NC/File_Upload_KSOA/poc.py``                                |
|                           | 用友 U8 OA test.jsp SQL注入漏洞                                       | ``poc/Yonyou_用友NC/SQLi_test_jsp/poc.py``                                   |
| Zabbix                    | Zabbix弱口令                                                       | ``poc/Zabbix/Weak_Pass/poc.py``                                            |
|                           | Zabbix SQL 注入漏洞(CVE-2016-10134)                                 | ``poc/Zabbix/SQLi_CVE_2016_10134/poc.py``                                  |
|                           | Zabbix未授权访问                                                     | ``poc/Zabbix/Auth_Bypass/poc.py``                                          |
| 禅道                        | 禅道8.2-9.2.1注入GetShell                                           | ``poc/Zentao_禅道/Getshell_test/poc.py``                                     |
| ZeroShell防火墙              | ZeroShell 3.9.0 远程命令执行漏洞                                        | ``poc/ZeroShell/RCE_kerbynet/poc.py``                                      |
| ZXOA_致翔OA                 | 致翔OA msglog.aspx SQL注入漏洞                                        | ``poc/ZXOA_致翔OA/SQLi_msglog_aspx/poc.py``                                  |
| Zyxel                     | Zyxel NBG2105身份验证绕过                                             | ``poc/Zyxel/Login_Pass_NBG2105/poc.py``                                    |
|                           | Zyxel 防火墙命令注入漏洞（CVE-2022-30525）                                 | ``poc/Zyxel/Command_Injection_CVE_2022_30525/poc.py``                      |

</details>

任何有兴趣为本项目作出贡献的同学都可以参与进来，详见[代码编写规范](https://github.com/bigblackhat/oFx/blob/main/docs/CODING.md)


<br>
<br>

## 🚔 法律免责声明

```
未经事先双方同意，使用oFx攻击目标是非法的。
oFx仅用于安全测试与研究目的

Usage of oFx for attacking targets without prior mutual consent is illegal.
oFx is for security testing purposes only
```

## 📜 Licenses

在原有协议LICENSE中追加以下免责声明。若与原有协议冲突均以免责声明为准。

本工具禁止进行未授权商业用途，禁止二次开发后进行未授权商业用途。

本工具仅面向合法授权的企业安全建设行为，在使用本工具进行检测时，您应确保该行为符合当地的法律法规，并且已经取得了足够的授权。

如您在使用本工具的过程中存在任何非法行为，您需自行承担相应后果，我们将不承担任何法律及连带责任。

在使用本工具前，请您务必审慎阅读、充分理解各条款内容，限制、免责条款或者其他涉及您重大权益的条款可能会以加粗、加下划线等形式提示您重点注意。
除非您已充分阅读、完全理解并接受本协议所有条款，否则，请您不要使用本工具。您的使用行为或者您以其他任何明示或者默示方式表示接受本协议的，即视为您已阅读并同意本协议的约束。

# 🎄 致谢清单

以下清单中的项目笔者都有参考或对笔者提供一定程度上的帮助，排名不分先后顺序，仅按照中文全拼首字母顺序排列

| 项目地址                                                                              |
|-----------------------------------------------------------------------------------|
| [AngelSword](https://github.com/Lucifer1993/AngelSword)                           |
| [edusrc_POC](https://github.com/Cl0udG0d/edusrc_POC)                              |
| [iak_ARL](https://github.com/nu0l/iak-ARL)                                        |
| [Landray-OA-Treexml-Rce](https://github.com/tangxiaofeng7/Landray-OA-Treexml-Rce) |
| [pocsuite3](https://github.com/knownsec/pocsuite3)                                |
| [sqlmap](https://github.com/sqlmapproject/sqlmap) sqlmap永远滴神                      |
| [vulhub](https://vulhub.org/)                                                     |
| [xray](https://github.com/chaitin/xray/) 抄作业，冲冲冲                                  |

# 😄 致谢JetBrains

``oFx``项目一直以来都是在 ``JetBrains`` 公司旗下的``PyCharm``集成开发环境中进行开发，基于 free JetBrains Open Source
license(s) 正版免费授权，在此表达我的谢意。

![JetBrains](img/008.png)

# 🛠 错误提交

如果您在使用oFx的过程中遇到了一些笔者写代码时没有考虑到的问题或没有测试到的错误，欢迎通过邮箱告知笔者(ronginus@qq.com)
或提Issues

提交BUG时，需要给到笔者触发错误的oFx命令，测试文件，您的运行环境（包括但不限于操作系统、python版本等），报错的字符串形式 ➕ 报错截图

~~都看到这儿了，点个star再走呗~~

![show](img/006.jpg)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

写一个好玩且能用的漏扫类的工具是我一直以来的执念，虽然离好用且完美还很远

感谢我的领导(id:``张真人``)给予我时间去自由探索，他也提出了很多前瞻性的idea（比如全量POC模式）

感谢我的同事(id:``hansi``)利用闲暇之余完成的POC贡献（见``changelog.md``）与前期使用体验的反馈

写oFx之前我的愿景是写一个概念验证工具，在hw等特殊时期网络上会充斥许许多多真真假假的漏洞，如何能仅根据语焉不详的利用描述就写出一个批量验证工具且具备无限的拓展性和复用性，根据这一需求，oFx的初代版本就这样诞生了，随着全量POC功能的实现，oFx更像一个专注于Nday检测的漏扫了，甚至可以用RCE漏洞去全网刷肉鸡(
我也确实是这样描述oFx的)，但我仍希望大家能用它来做安全研究工作

接下来的时光我将会沉浸于代码审计的工作，oFx的架构不会有明显的调整，但POC应该会持续的不定期的更新

出于对萌新使用体验的考虑，我会将核心的输出信息更为中文

感谢读到这里，祝好祝顺

``访客数统计``  
![Visitor Count](https://profile-counter.glitch.me/bigblackhat-oFx/count.svg)  
``Star统计``  
[![Stargazers over time](https://starchart.cc/bigblackhat/oFx.svg)](https://starchart.cc/bigblackhat/oFx)