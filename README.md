# 🖥️ Softmax Classification for MNIST Digit Recognition

## 🧾 Project Overview  
This project demonstrates a **Softmax Classifier** for handwritten digit recognition using PyTorch, trained on the MNIST dataset. It includes implementations of Softmax classification in one dimension and a multi-class classification for handwritten digits.

## ✨ Key Features  
- 🧮 Softmax classification implementation  
- 📂 MNIST dataset loading and preprocessing  
- 🚀 Model training and evaluation  
- 📊 Visualization of model parameters and results  

## ⚙️ Prerequisites  
- 🐍 Python 3.8+  
- 🔥 PyTorch  
- 📉 Matplotlib  
- 🖼️ Torchvision  

## 📥 Installation  
```bash  
git clone https://github.com/muizz8/Softmax-Classification-for-MNIST-Digit-Recognition.git  
cd Softmax-Classification-for-MNIST-Digit-Recognition  
pip install torch torchvision matplotlib  
```  

## 📂 Project Structure  
- `softmax_classifier.ipynb`: Main implementation of Softmax classification  

## 🧩 Key Components  
### 🧠 Softmax Classifier  
- Uses a **linear layer** for classification  
- Applies **Cross-Entropy Loss**  
- **Stochastic Gradient Descent (SGD)** optimizer  

### 📊 Visualization  
- Parameter visualization of trained model  
- Loss and accuracy tracking  
- Misclassification analysis  

## 🚀 Usage  
```python  
# Train the model  
model = SoftMax(input_size=784, output_size=10)  
train_model(epochs=10)  

# Evaluate and visualize  
PlotParameters(model)  
```  

## 📈 Results  
- ✅ Trained on MNIST dataset  
- 📉 Visualizes model learning process  
- ❌ Shows misclassified and correctly classified samples  

## 🙌 Acknowledgments  
- 📚 **MNIST Dataset**  
- 🔥 **PyTorch Team**
