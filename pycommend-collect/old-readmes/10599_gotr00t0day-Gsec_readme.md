![Gsec](https://media.discordapp.net/attachments/700021958896648242/1052049111978889226/GSec_Banner_Transparente.png)
<h4 align="center">Web Security Scanner &amp; Exploitation.</h4>
<h4 align="center">Based on custom vulnerability scanners &amp; Nuclei</h4>
<h4 align="center">

![Python Version](https://img.shields.io/badge/python-3.11.3-green)
![Issues](https://img.shields.io/github/issues/gotr00t0day/Gsec)
![Stars](https://img.shields.io/github/stars/gotr00t0day/Gsec)
![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fgotr00t0day)
</h4>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#keys">Keys</a> •
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#Keywords">KeyWords</a> •
  <a href="https://discord.gg/59cKfqNNHq">Join Discord</a>

</p>

<hr>

## Features

   * Passive Scan
     - Find assets with shodan
     - RapidDNS to get subdomains
     - Certsh to enumerate subdomains
     - DNS enumeration
     - Waybackurls to fetch old links
     - Find domains belonging to your target
   
   * Normal / Agressive Scan
     - Domain http code
     - Web port scanning
     - Server information
     - HTTP security header scanner
     - CMS security identifier / misconfiguration scanner
     - Technology scanner 
     - Programming Language check
     - Path Traversal scan
     - Web Crawler
     - OS detection
     - Nuclei vulnerability scanning
     - SSRF, XSS, Host header injection and Cors Misconfiguration Scanners.

<hr>

## Installation

Make sure you have GoLang installed, with out it you won't be able to install nuclei.

```bash

git clone https://github.com/gotr00t0day/Gsec.git

cd Gsec

pip3 install -r requirements.txt

# Make sure that nuclei-templates is cloned in the / directory. Gsec fetches the templates from ~/nuclei-templates
python3 install.py

```

## Keys

```bash

Gsec will fetch the shodan API key from the core directory, the passive recon script supports scanning with shodan,
please save your shodan key in core/.shodan for the scan to be able to work.


```

## OUTPUT

```bash

Some outputs that are too large will be saved in a file in the output folder / directory.


```

## Usage

```bash
# normal (passive and aggresive scans)

python3 gsec.py -t https://domain.com

# Passive Recon

python3 gsec.py -t https://domain.com --passive_recon

# Ultimate Scan (Scan for High and Severe CVEs and Vulnerabilities with nuclei)

python3 gsec.py --ultimatescan https://target.com

```

# Anonimity

## ProxyChains

You can use Proxychains with tor for anonimity.

```bash

proxychains -q python3 gsec.py -t https://target.com

```

## Keywords

If Gsec finds a vulnerability and it has the *POSSIBLE!* keyword in the output that means it could be a false positive and you need to manually test the vulnerability to make sure it's actually vulnerable.

## Coming Soon...

I'm working on adding proxy support for Gsec, it will be added in future releases.

## Issues

In python3.10+ you might get an SSL error while running Gsec. To fix this issue just ``` pip3 install ceritifi ``` and then do ```/Applications/Python\ 3.10/Install\ Certificates.command``` and the issue will be fixed.
