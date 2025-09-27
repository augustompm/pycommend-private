## blackjump

[中文](https://github.com/tarimoe/blackjump/) | [English](https://github.com/tarimoe/blackjump/blob/main/README_en.md)

> 免责声明: 本工具仅面向合法授权的企业安全建设行为，在使用本工具进行检测时，<b>您应确保该行为符合当地的法律法规，并且已经取得了足够的授权。请勿对非授权目标使用</b>。
> 
> 如您在使用本工具的过程中存在任何非法行为，<b>您需自行承担相应后果，我们将不承担任何法律及连带责任</b>。


JumpServer 堡垒机综合漏洞利用
- [x] 未授权任意用户密码重置 (CVE-2023-42820)
- [x] 未授权一键下载所有操作录像 (CVE-2023-42442)
- [x] 未授权任意命令执行漏洞 (RCE 2021)

## 安装
```bash
python3 -m pip install -r requirements.txt
```

## 使用指南
+ CVE-2023-42820: 如果知道目标的用户名和邮箱可以指定 `--user` 和 `--email` 参数
```bash
python3 blackjump.py reset https://vulerability
```
![img.png](img/img.png)

+ CVE-2023-42442: `output/` 目录下的 `<uuid4>.tar` 文件扔进 <u>[jumpserver播放器播放即可](https://github.com/jumpserver/VideoPlayer/releases)</u> 
```bash
python3 blackjump.py dump https://vulerability
```
![img_1.png](img/img_1.png)e

+ RCE
```shell
python3 blackjump.py rce http(s)://vulerability
```
![img.png](img/img_2.png)

+ 帮助
```bash
python3 blackjump.py {reset,dump,rce} -h
```

## 参考
1. https://github.com/Veraxy00/Jumpserver-EXP (RCE 2021 漏洞在其基础上优化部分情况命令执行失败或获取不到资产问题)
