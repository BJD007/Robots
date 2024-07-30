## Step 1: Set Up Your Environment
- Install Required Software:
    - Python: Ensure you have Python 3.7 or later installed.
    - PyTorch: Install PyTorch using pip install torch torchvision.
    - TensorFlow: Install TensorFlow using pip install tensorflow.
    - ROS-2: Follow the official ROS-2 installation guide for Ubuntu. https://docs.ros.org/en/foxy/Installation/Alternatives/Ubuntu-Development-Setup.html
    - TensorRT: Install TensorRT by following the NVIDIA TensorRT installation guide https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

- To check if the above is already installed
    - Python: python3 --version (This should show Python 3.7 or later)
    - PyTorch: Install PyTorch using pip install torch torchvision.(If PyTorch is installed, this will print its version)
    - TensorFlow: python3 -c "import tensorflow as tf; print(tf.__version__)"
    - ROS-2: ros2 --version

    
- Create a Virtual Environment:
    - python3 -m venv perception_env
    - source perception_env/bin/activate

- Install Additional Dependencies:
    - pip install numpy opencv-python matplotlib

## Step 2: Data Collection and Preprocessing
    - We'll use the KITTI dataset for this example. 
    - Download the dataset and preprocess the images.
       
        * Download and Extract the Dataset

            * Create a directory for the dataset
                *  mkdir -p ~/datasets/kitti

        * Navigate to the dataset directory
            * cd ~/datasets/kitti

        * Download the dataset (example for object detection)
            * wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
            * wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

        * Extract the downloaded files
            * unzip data_object_image_2.zip
            * unzip data_object_label_2.zip

## Step 2: Model building
    - Pytorch Framework
        - yolo_pytorch.py
        - ssd_pytorch.py
        - unet_pytorch.py
        - ViT_pytorch.py
    - TensorFlow Framework
        - yolo_tg.py
        - ssd_tf.py
        - unet_tf.py
        - ViT_tf.py

## Step 3: Data Collection, PreProcessing + Model Selection and Training
- Data Collection, Preprocessing and Model Selection Script
    - datacollectionNpreprocessing.py
    - Datasets: Download datasets like KITTI, NuScenes, or Cityscapes.
    - Preprocessing: Use Python libraries such as OpenCV and NumPy for data augmentation, normalization, and splitting.

## Step 4: Model Optimization
- Model Selection and Training
    - modelselectionNtraining.py

## Step 5: Integration with ROS-2 and TensorRT
- ROS-2 Node Script
    - perception_node.py
    - image_publisher.py
    - detection_visualize.py

- This setup will allow you to see the inference results both through the OpenCV window created by the perception node and through RViz (if you choose to use it).
    - Terminal 1: Run image_publisher.py
    - Terminal 2: Run perception_node.py
    - Terminal 3: Run rviz2
    - Terminal 4: detection_visualize.py

## Step 6: Testing and Validation
- Use simulation tools like Gazebo or CARLA to test the integrated system. If possible, deploy the system on a robotic platform and test it in real-world conditions.
    - carla_test.py
    - realworld_test.py


## Step 7: Testing and Validation
- Convert the trained model to TensorRT for deployment on NVIDIA GPUs, ensuring real-time performance.
    - yolo_tensorRT.py
    - ssd_tensorRT.py
    - unet_tensorRT.py
    - ViT_tensor_RT.py

## Step 8: After converting the models to TensorRT, load and use them for inference
- tensorRT_inference.py
