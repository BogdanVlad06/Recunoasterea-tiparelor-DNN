### Download RO PDF File
- [PDF Version](document.pdf)

---

### **Pattern Recognition Using Neural Networks**

**Specialized Project Romanian high school**

**Field: Theoretical**

**Profile: Real**

**Specialization: Mathematics – Computer Science, Intensive Computer Science**

Class: **12th Grade D**

---

### **Contents**

- [*Introduction*](#_introduction)
  - [*Problem Statement*](#_problem-statement)
  - [*Approach*](#_approach)
- [*Motivation*](#_motivation)
- [*Project Description*](#_project-description)
  - [**A. Artificial Intelligence Component**](#_ai-component)
    - [**I. Composition and Structure**](#_composition)
    - [**II. Training Mechanism**](#_training-mechanism)
    - [**III. Dataset and Improvements**](#_dataset)
  - [**B. Web Interface Component**](#_web-interface)
- [*Programming Language and Tools*](#_tools)
- [*Interface Instructions*](#_interface-instructions)
- [*Bibliography*](#_bibliography)

---

### **Introduction**

#### **Problem Statement**
Handwritten digit recognition is a challenge due to the natural variations in individual handwriting styles. While humans easily adapt to these differences, the question arises: can we create a program capable of simulating the human brain's ability to process information?

#### **Approach**
Neural networks emerge as a solution, replicating this process to offer efficient and precise methods, overcoming the limitations of traditional algorithms.

---

### **Motivation**
In an era defined by rapid technological advancements, *artificial intelligence (AI)* has become a cornerstone across various sectors. This project aligns with this global trend and reflects a personal interest in exploring innovative methods to solve complex problems. By creating a neural network, I sought to integrate theoretical knowledge from mathematics and computer science into a practical project.

The foundation of this concept lies in the biological neuron, the basic unit of the human nervous system. By abstracting the neuron's ability to receive, process, and transmit information, artificial neurons have been developed as powerful mathematical models capable of data analysis, learning, and automated decision-making.

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

### **Interface Instructions**

- **Canvas**:
  - Draw digits using the mouse.
  - Adjust brush size using the slider.
- **Network Visualization**:
  - View neuron activations and weight connections.
- **Buttons**:
  - Reset, Test, Save, Load, Random Test Image, Reinitialize Network.

---

### **Bibliography**

- [Neural Networks by 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks)
- [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
