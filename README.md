
# MEGA (.pytorch) Installation and Demo

These instructions guide you through installing the `mega.pytorch` repository and running the demo.

**Prerequisite:** You must have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

---

## 🛠️ Installation


You can find the updated files in the modified_files folder of this project, you can just copy/paste them to substitute the original ones, but down below you can find also the step-by-step instructions to do it manually.

### Step 1: Environment Setup & Core Dependencies

1.  Create and activate a new conda environment.

    ```bash
    # first, make sure that your conda is setup properly with the right environment
    # for that, check that `which conda`, `which pip` and `which python` points to the
    # right path. From a clean conda env, this is what you need to do

    conda create --name MEGA -y python=3.7
    source activate MEGA

    # this installs the right pip and dependencies for the fresh python
    conda install ipython pip
    ```

2.  Install MEGA and COCO API dependencies.

    ```bash
    pip install ninja yacs cython matplotlib tqdm opencv-python scipy
    ```

3.  Install PyTorch (instructions for CUDA 10.0).

    ```bash
    # follow PyTorch installation in https://pytorch.org/get-started/locally/
    # we give the instructions for CUDA 10.0
    conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
    ```

4.  Set an installation directory variable for convenience.

    ```bash
    export INSTALL_DIR=$PWD
    ```

5.  Install `pycocotools`.

    ```bash
    cd $INSTALL_DIR
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext install
    ```

6.  Install `cityscapesScripts`.

    ```bash
    cd $INSTALL_DIR
    git clone https://github.com/mcordts/cityscapesScripts.git
    cd cityscapesScripts/
    python setup.py build_ext install
    ```

7.  Clone the `apex` repository.

    ```bash
    cd $INSTALL_DIR
    git clone https://github.com/NVIDIA/apex.git
    cd apex
    ```

### Step 2: Patch and Install Apex & mega.pytorch


1.  **IMPORTANT:** Before installing `apex`, you must manually edit its `setup.py` file.
    * Open `$INSTALL_DIR/apex/setup.py`.
    * Find this line:
        ```python
        parallel: int | None = none
        ```
    * Change it to:
        ```python
        from typing import Optional
        parallel: Optional[int] = None
        ```
      
    * or:
        ```python
        parallel: int = None
        ```

2.  Now, build and install `apex` (make sure you are still in the `$INSTALL_DIR/apex` directory).

    ```bash
    python setup.py build_ext install
    ```

3.  Clone and install `mega.pytorch`.

    ```bash
    cd $INSTALL_DIR
    git clone https://github.com/AdondeBoy/DLVSP-Lab-2
    cd DLVSP-Lab-2/mega.pytorch

    # the following will install the lib with
    # symbolic links, so that you can modify
    # the files if you want and won't need to
    # re-build it
    python setup.py build develop
    ```

4.  Install a specific Pillow version.

    ```bash
    pip install 'pillow<7.0.0'
    ```

5.  Unset the environment variable.

    ```bash
    unset INSTALL_DIR
    ```

---

## 🚀 Running the Demo

Before you can run the demo script, you need to apply a few manual patches to the code. The modifications in this step are already done in the current repository, but just in case you would like to recreate this setup from the original version, here are the instructions.

### Step 1: Patch Apex Dependencies

You need to remove references to `apex.amp` from several files.

**1.1: Edit `mega.pytorch/mega_core/layers/nms.py`**
* **Delete** this line:
    ```python
    from apex import amp
    ```
* **Change** this line:
    ```python
    nms = amp.float_function(_C.nms)
    ```
* **To this:**
    ```python
    nms = _C.nms
    ```

**1.2: Edit `mega.pytorch/mega_core/layers/roi_align.py`**
* **Delete** this line:
    ```python
    from apex import amp
    ```
* **Delete** this line (it's a decorator right above a function):
    ```python
    @amp.float_function
    ```

**1.3: Edit `mega.pytorch/mega_core/layers/roi_pool.py`**
* **Delete** this line:
    ```python
    from apex import amp
    ```
* **Delete** this line (it's a decorator):
    ```python
    @amp.float_function
    ```

### Step 2: Patch Demo Predictor

1.  Open `mega.pytorch/demo/predictor.py`.
2.  **Change** this line:
    ```python
    image, s, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2
    ```
3.  **To this** (casting `x` and `y` to `int`):
    ```python
    image, s, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2
    ```

### Step 3: Run the Demo Scripts

1.  Make sure you are in the `mega.pytorch` directory in your terminal.

2. If you want to run the demo with a folder full of images, use the following commands:

   1. **To run the basic demo:**
       ```bash
       python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".IMAGES_SUFFIX" --visualize-path path_to_your_image_folder/image_folder --output-folder path_to_your_output_folder
       ```

   2. **To run the MEGA model demo:**
       ```bash
       python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".IMAGES_SUFFIX" --visualize-path path_to_your_image_folder/image_folder --output-folder path_to_your_output_folder
         ```
3. If you want to run the demo with a video, use the following commands:

   1. **To run the basic demo:**
       ```bash
       python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".YOUR_SUFFIX" --visualize-path path_to_your_video.YOUR_SUFFIX --output-folder path_to_your_output_folder --video
       ```

   2. **To run the MEGA model demo:**
       ```bash
       python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".YOUR_SUFFIX" --visualize-path path_to_your_video.YOUR_SUFFIX --output-folder path_to_your_output_folder --video
       ```
4. You can add **--output-video** if you want to save the output as a video file and not as independent frames.