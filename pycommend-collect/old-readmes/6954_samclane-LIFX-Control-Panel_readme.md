## Development is now being continued on the new [Mantle](https://github.com/samclane/mantle) project

# LIFX-Control-Panel

[![codecov](https://codecov.io/gh/samclane/LIFX-Control-Panel/branch/master/graph/badge.svg?token=GLAxucmOo6)](https://codecov.io/gh/samclane/LIFX-Control-Panel)
![Smoke Build And Test](https://github.com/samclane/LIFX-Control-Panel/actions/workflows/main.yml/badge.svg?event=push&branch=master)

<img align="right" width="120" height="120" title="LIFX-Control-Panel Logo" src="./res/lifx-animated-logo.gif">
     
LIFX-Control-Panel is an open source application for controlling your LIFX brand lights. It integrates simple features, 
such as monitoring and changing bulb color, with more advanced ones, like:
 
 * Average Screen Color
 * Color Eyedropper
 * Custom color palette
 * Keybindings

<p align="center">
  <img src="./res/screenshot.png" alt="Screenshot" width="306" height=731>
</p>

#### Color Averaging Demo (Click for Video):

[![Avg Test Youtube](https://img.youtube.com/vi/C-jZISM9MC0/0.jpg)](https://youtu.be/C-jZISM9MC0)

The application uses a fork of [mclarkk's](https://github.com/mclarkk)'s [lifxlan](https://github.com/mclarkk/lifxlan) module to
discover and send commands to the lights.

[The fork can be found here.](https://github.com/samclane/lifxlan)

# Quick Start

There are ~~2~~ **3** ways to install:

1. Go over to [releases](https://github.com/samclane/LIFX-Control-Panel/releases) and download the latest `.exe` file.

2. Run `pip install lifx-control-panel`. To start run `python -m lifx_control_panel`.

Starting the program takes a moment, as it first must scan your LAN for any LIFX devices.

# Running the source code

You can now install through PyPI, by running `pip install lifx-control-panel`. This will automatically install dependencies.

To manually install the dependencies, run `pip install -r requirements.txt`. 

Due to some initial PyCharm cruft, the environment paths are a bit messed up. 
- The main script path is:
  - `..\LIFX-Control-Panel\lifx_control_panel\__main__.pyw`
- The Working Directory is:
  - `..\LIFX-Control-Panel\lifx_control_panel`
- Additionally, the `Add content roots to PYTHONPATH` and `Add source roots to PYTHONPATH` boxes are checked
  - I haven't been able to reproduce this in VSCode, yet.
# Building

LIFX-Control-Panel uses PyInstaller. After downloading the repository, open a command window in the `LIFX-Control-Panel`
directory, and run `pyinstaller __main__.pyw`. This should initialize the necessary file structure to build the project.

As admin, run `pip install -r requirements.txt` to install project dependencies. Note that pyaudio requires the use of pipwin to install - use `pip install pipwin` to obtain pipwin then install pyaudio using `pipwin install pyaudio`

To build the project, simply open a terminal in the same folder and run `build_all.bat` in the command prompt. It will
call `pyinstaller` on `build_all.spec`. This should generate `.exe` files in the `/dist`
folder of the project for each of the 3 specs:

- `main`
  - This is the file that is used to build the main binary. The console, as well as verbose logging methods, are disabled.
- `debug`
  - This spec file enables the console to run in the background, as well as verbose logging.
- `demo`
  - The demo mode simulates adding several "dummy" lights to the LAN, allowing the software to be demonstrated on networks
    that do not have any LIFX devices on them.

If you need help using PyInstaller, more instructions are located [here](https://pythonhosted.org/PyInstaller/usage.html).

# Testing progress

I have currently only tested on the following operating systems:

- Windows 10

and on the following LIFX devices:

- LIFX A19 Firmware v2.76
- LIFX A13 Firmware v2.76
- LIFX Z Firmware v1.22
- LIFX Mini White Firmware v3.41
- LIFX Beam

I've tried to test on the following operating systems:
- MacOS X
- Fedora Linux

However, the biggest hurdle seems to be the `tk` GUI library, which is not supported on MacOS X, and requires
extra library installations on Linux.

# Feedback

If you have any comments or concerns, please feel free to make a post on the [Issues page](https://github.com/samclane/LIFX-Control-Panel/issues).

If you enjoy LIFX-Control-Panel, please Like and leave a review on [AlternativeTo](https://alternativeto.net/software/lifx-control-panel/).

### NEW

[Join our Discord Server](https://discord.gg/Wse9jX94Vq)

# Donate

LIFX-Control-Panel will always be free and open source. However, if you appreciate the work I'm doing and would like to
contribute financially, you can donate below. Thanks for your support!

<a href='https://ko-fi.com/J3J8LZKP' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi3.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.me/sawyermclane)
