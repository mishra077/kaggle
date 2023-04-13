# Kaggle

## All competition which I've completed or ongoing will be updated here!
</br>

### Installation

</br>

#### For Ubuntu/RedOS/CentOS

</br>
For PIP

- Python installation
    ```sh
    sudo apt-get update
    sudo at-get install python3.8
    ```
- To check the version:
    ```sh
    python --version
    ```
- Create conda environment
    ```sh
    python3.8 -m venv <env-name>
    source <env-name>/bin/activate
    ```
- Install packages from requirements.txt
    ```sh
    pip install -r requirements.txt
    ```

Anaconda installation

- Download Installer
    ```sh
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
    sha256sum Anaconda3-2021.05-Linux-x86_64.sh
    bash Anaconda3-2021.05-Linux-x86_64.sh
    ```
- Add Anaconda to your System's PATH variable
    ```sh
    vi ~/.bashrc
    # at the end of the file
    export PATH = /home/<user-name>/anaconda3/bin:$PATH"
    source ~/.bashrc
    ```
- Create conda environment
    ```sh
    conda create -n <env-name> python=3.8
    conda activate <env-name>
    ```
- Install packages from conda
    ```sh
    conda install --file requirements.txt
    ```
