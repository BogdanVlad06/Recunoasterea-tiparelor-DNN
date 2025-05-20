### Download RO PDF File
- [PDF Version](document.pdf)

---

### **Pattern Recognition Using Neural Networks**

**High School Graduation Project**

**Field: Theoretical Studies**

**Specialization: Mathematics & Computer Science**

Class: **12th Grade**

---

### **Contents**

- [*Introduction*](#introduction)
  - [*Problem Statement*](#problem-statement)
  - [*Approach*](#approach)
- [*Motivation*](#motivation)
- [*Project Description*](#project-description)
  - [**A. Artificial Intelligence Component**](#a-artificial-intelligence-component)
    - [**I. Composition and Structure**](#i-composition-and-structure)
    - [**II. Training Mechanism**](#ii-training-mechanism)
    - [**III. Dataset and Improvements**](#iii-dataset-and-improvements)
  - [**B. Web Interface Component**](#b-web-interface-component)
- [*Programming Language and Tools*](#programming-language-and-tools)
- [*User Guide*](#user-guide)
- [*References*](#references)

---

### **Introduction**

#### **Problem Statement**
Recognizing handwritten digits is challenging due to the wide variability in personal handwriting styles. While humans adapt to these differences intuitively, the core question is: Can we develop a system that mimics the brain’s capacity to interpret such input?

#### **Approach**
Neural networks emerge as a solution, replicating this process to offer efficient and precise methods, overcoming the limitations of traditional algorithms.

---

### **Motivation**
As artificial intelligence becomes increasingly influential across industries, this project reflects both a global trend and a personal interest in innovative problem-solving. My goal was to apply theoretical knowledge from mathematics and computer science to build a practical and interactive AI system.

Inspired by the biological neuron — the fundamental unit of the nervous system — artificial neurons replicate the ability to receive, process, and transmit information. This abstraction has proven effective for tasks such as learning, classification, and automated decision-making.

---

### **Project Description**

This project involves building the architecture of a **neural network (NN) from scratch** to recognize handwritten digits. While digit recognition is typically associated with convolutional neural networks (CNNs), I opted for a feed-forward architecture to illustrate the fundamental principles of training a neural network in a transparent manner. The network was integrated into an **intuitive web interface** that allows users to test the model's functionality online. The project has two primary components: the **AI model** and the **web interface**.

#### **A. Artificial Intelligence Component**

##### **I. Composition and Structure**

- **The Neuron**: At its core, a neuron is a simple mathematical function with parameters: weights (`w`) and a bias (`b`). It receives input values (`A`), computes a weighted sum (`z`), and applies an activation function to produce an output.
- **Layers**: A layer is a group of neurons working in parallel. Each layer processes inputs from the previous layer and passes its outputs to the next. I used matrix operations to streamline calculations, with formulas such as: 
  
  \[ Z^{[L]} = W^{[L]} \cdot A^{[L-1]} + B^{[L]} \]

- **Activation Function**: I used the **ReLU (Rectified Linear Unit)** function defined as:
  \[ ReLU(x) = \max(0, x) \]

- **Network Structure**:
  - Input Layer: Stores 784 activations (for 28×28 images).
  - Hidden Layers: Two layers with 16 neurons each for processing data.
  - Output Layer: Contains 10 neurons (one for each digit, 0–9). A **softmax** function was applied here to compute probabilities.

##### **II. Training Mechanism**

1. **Forward Propagation**:
   - Input data is passed through the network, and activations are calculated layer by layer.
2. **Error Calculation**:
   - The error is computed using a loss function. I used **Cross-Entropy Loss**, which measures the difference between predicted probabilities and actual labels.
3. **Backpropagation**:
   - Gradients of the error with respect to each parameter are calculated using the chain rule and derivatives. These gradients indicate how to adjust parameters to minimize error.
4. **Parameter Update**:
   - Parameters are updated using gradient descent, scaled by a learning rate (`η`).

##### **III. Dataset and Improvements**

- **Dataset**: The Modified National Institute of Standards and Technology (MNIST) database was used, consisting of 60,000 training images and 10,000 test images (28×28 pixels each).
- **Data Preprocessing**:
  - Flattened images into 1D vectors of 784 values.
  - Normalized pixel values to a range of 0–1.
  - Retained labels for error calculation.
- **Improvements**:
  - Mini-batch training with Stochastic Gradient Descent.
  - Adaptive learning rate adjustment.
  - Parameter initialization using He Initialization.

#### **B. Web Interface Component**

The web interface serves as the primary user interaction point, allowing real-time testing of the neural network. Built with JavaScript and the **p5.js** library, the interface includes:

- **Drawing Canvas**: A 28×28 grid where users can draw digits.
- **Visualization**: Displays the network's structure, including weights and activations.
- **Controls**:
  - Reset Canvas
  - Test
  - Save/Load Model
  - Random Test Image
  - Reinitialize Network

---

### **Programming Language and Tools**

- **Languages**: HTML, CSS, JavaScript
- **Libraries**: p5.js, math.js, FileSaver.js
- **Tools**: Visual Studio Code with Git integration

---

### **User Guide**

- **Canvas**:
  - Draw digits using the mouse.
  - Adjust brush size using the slider.
- **Network Visualization**:
  - View neuron activations and weight connections.
- **Buttons**:
  - Reset, Test, Save, Load, Random Test Image, Reinitialize Network.

---

### **References**

- [Neural Networks by 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks)
- [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
