# Deep Learning for Traffic Signs Regconition 
deepLearningTrafficSigns is a machine learning project designed to recognize and classify traffic signs using convolutional neural networks (CNNs). This application aims to identify traffic signs from images captured via a webcam, demonstrating practical applications of deep learning in computer vision.

# Features
- Traffic Sign Recognition: Uses a trained CNN model to identify traffic signs in real-time.
- Real-time Detection: Integrates with a webcam to provide live traffic sign detection.
- Comprehensive Training Dataset: Utilizes a diverse dataset of traffic sign images for model training.
Technologies Used
- Programming Language: Python
- Libraries: OpenCV, TensorFlow, Keras, NumPy
- Machine Learning: Convolutional Neural Networks (CNNs)

## Installation
To set up the project locally, follow these steps:

Clone the repository:
```
git clone https://github.com/anthony12031/deepLearningTrafficSigns.git
cd deepLearningTrafficSigns
```

## Install dependencies:

Ensure you have Python installed. Then, install the required libraries:

```
pip install -r requirements.txt
Download the dataset:
```

Download a traffic signs dataset (e.g., German Traffic Sign Recognition Benchmark) and place it in the data directory.

## Usage
Training the Model:

To train the model using the dataset, run the following command:
```
python train.py
This will train the CNN model and save the trained model to the models directory.
```

## Running the Application:

To start the traffic sign recognition application, run:

```
python recognize.py
```
This will launch the application, open a webcam feed, and start detecting traffic signs in real-time.

## Project Structure
- train.py: Script for training the CNN model.
- recognize.py: Script for running the traffic sign recognition application.
- models/: Directory to save and load trained models.
- data/: Directory to store the traffic signs dataset.
- requirements.txt: File listing the required libraries and dependencies.
  
## Contributing
We welcome contributions from the community. To contribute:

## Fork the repository.
Create a new branch with a descriptive name.
Make your changes and commit them with clear messages.
Push your changes to your fork.
Submit a pull request with a detailed explanation of your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For questions or feedback, please reach out to Anthony Jason Vargas Sepúlveda.
