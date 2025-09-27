<div align="center" style="background-color: #ffcccc; border: 2px solid #ff0000; padding: 15px; margin: 20px 0; border-radius: 5px;">
  <h2 style="color: #ff0000; text-align: center; margin-top: 0;">⚠️ REPOSITORY ARCHIVED ⚠️</h2>
  <p style="font-size: 16px; text-align: center;">
    This repository is no longer actively maintained.<br>
    It has been archived for historical reference.<br>
    Thank you to all contributors and users of this project.<br>
    Last updated: Oct 21, 2023
  </p>
  <hr style="border-top: 1px solid #ff0000; margin: 15px 0;">
  <p></p>
</div>


<p align="center">

  <img src="https://github.com/kaboussi/Botflix/assets/65725801/a0f2db38-fe3c-48af-b63c-30b6aabcd288" />
</p>

<h2 align='center'> Botflix </h2>

<div align='center'>
<a href="https://aur.archlinux.org/packages/botflix-git">
	
![AUR version](https://img.shields.io/aur/version/botflix-git)

</a>
</div>

Botflix is a Python scrapping CLI that combines [scrapy](https://scrapy.org) and [webtorrent](https://github.com/webtorrent/webtorrent-cli) in one command for streaming movies from your terminal.

## Installation:
#### For ArchLinux or Arch-based distro:
You can install it manually from [AUR repository](https://aur.archlinux.org/packages/botflix-git) or use a aur helper like `yay` or `paru`.
```bash
yay -Syu botflix-git
```
To run
```bash
botflix
```
`botflix` is the replacement for the command prefix `python3 main.py` in this readme. for example `python3 main.py config "vlc"` will be `botflix config "vlc"`, the final step after installation.
#### For others
> Botflix is written in Python, and it depends on [webtorrent](https://github.com/webtorrent/webtorrent-cli), [NodeJS](https://nodejs.org) and [npm](https://www.npmjs.com)

1. install NodeJS on your machine ([read mode](https://nodejs.org/en/download/)).
```bash
node --version
v17.9.0 #or higher
```
2. make sure that you have npm already installed ([read more](https://docs.npmjs.com/cli/v7/configuring-npm/install)).
```bash
npm --version
8.8.0 # or higher
```
3. now let's install webtorrent ([read more](https://github.com/webtorrent/webtorrent-cli)).
```bash
npm install webtorrent-cli -g
webtorrent --version # 4.0.4 (1.8.16)
```
_Note: if not installed try with sudo privileges._

4. clone the repo in your local machine.
```bash
    git clone https://github.com/kaboussi/Botflix && cd Botflix
```
![clone results](.github/clone.png)

5. create a virtual environment.
* Beginner Windows users who couldn't set up the [virtualenv] check this doc.
* Unix Users
```bash
python3 -m venv venv && source venv/bin/activate
```
![creating virtualenv](.github/virtualenv.png)  
**Note**: on Debian/Ubuntu systems you will first need to install venv using 
```bash
sudo apt install python3-venv
```

6. Install necessary packages.
```bash
pip install -r requirements.txt
```
![install packages](.github/pipintsall.png)

## Usage:
* First you need to set up a default player.
> Note that only [vlc](https://www.videolan.org/vlc/) and [mpv](https://mpv.io/) are supported. <br>
> If you are a Windows user, make sure you add your player to the PATH. [read more](https://www.vlchelp.com/add-vlc-command-prompt-windows/)
```bash
python3 main.py config "vlc"
```
![config](.github/config.png)
* If you want to get top movies:
```bash
python3 main.py top
```
* If you want to watch a TV Series:
```bash
python3 main.py serie
```
* If you want to search for a specific movie (_"[red notice](https://www.imdb.com/title/tt7991608/)"_ for example):
```bash
python3 main.py search
What movie are you looking for? red notice
```
![table of search](.github/table_of_movies.png "table of search")
* To start watching you can just type the number of the movie in the table.

## Contributing:
Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch `git checkout -b feature/AmazingFeature`
3. Commit your Changes `git commit -m 'Add some AmazingFeature`
4. Push to the Branch `git push origin feature/AmazingFeature`
Open a Pull Request.

## Contributors❤:
<div align="center">
	<a href="https://github.com/kaboussi/Botflix/graphs/contributors">
  	<img src="https://contrib.rocks/image?repo=kaboussi/Botflix" />
	</a>
</div>

## License:
[MIT](https://mit-license.org/)<br>
[DISCLAIMER](https://www.disclaimertemplate.net/live.php?token=xyytrgo4QtkLMNCB6LEIO6Q39YDFyhu2)



<!-- Links -->
[virtualenv]: https://github.com/kaboussi/Botflix/blob/main/docs/windowsVenv.md
