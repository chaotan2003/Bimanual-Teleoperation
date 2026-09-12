#!/bin/bash

# Check the operating system
OS_NAME=$(uname -s)
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION=$(lsb_release -rs)
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION=$VERSION_ID
    fi
    if [[ "$OS_VERSION" != "22.04" ]]; then
        echo "Warning: This script has only been tested on Ubuntu 22.04"
        echo "Your system is running Ubuntu $OS_VERSION."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

    # Install the required packages
    rm -rf dependencies
    mkdir dependencies
    cd dependencies

    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
    cd XRoboToolkit-PC-Service-Pybind
    bash setup_ubuntu.sh

    cd ..
    git clone https://github.com/zhigenzhao/R5.git
    cd R5
    git checkout dev/python_pkg
    cd py/ARX_R5_python/
    pip install .

    cd ../../../..

    pip install -e . || { echo "Failed to install xrobotoolkit_teleop with pip"; exit 1; }


    echo -e "\n"
    echo -e "[INFO] xrobotoolkit_teleop is installed in conda environment '$ENV_NAME'.\n"
    echo -e "\n"
