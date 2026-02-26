# Resume-Classification-System-using-DistilBERT
A multi-class Resume Classification system built using a pretrained Transformer model (DistilBERT) and fine-tuned on a custom resume dataset.
This project demonstrates how to build an industry-level NLP classification pipeline using transfer learning.

##  Project Overview

Recruiters often receive hundreds of resumes for different technical roles.  
This system automatically classifies resumes into job categories using a fine-tuned Transformer model.

Instead of training a neural network from scratch, this project leverages a pretrained language model and adapts it to a resume classification task.



##  Model Architecture

- **Base Model:** `distilbert-base-uncased`
- **Framework:** Hugging Face Transformers
- **Backend:** PyTorch
- **Training Environment:** Google Colab (GPU enabled)
- **Task Type:** Multi-class Text Classification

DistilBERT was chosen because it is:

- Lightweight
- Faster than BERT
- Memory efficient
- Suitable for Google Colab GPU training



##  Dataset

The dataset contains resume samples labeled into multiple job categories.

### Dataset Structure

| resume_text | category |
|-------------|----------|
| Resume content... | Data Science |
| Resume content... | Web Development |

### Categories Included

- Data Science  
- Web Development  
- Mobile Development  
- DevOps  
- Cybersecurity  
- UI/UX Design  
- Cloud Engineering  



##  Training Pipeline

The model training process follows these steps:

1. Load CSV dataset  
2. Encode text labels into numeric format  
3. Split into train/validation sets  
4. Tokenize text using DistilBERT tokenizer  
5. Fine-tune pretrained model  
6. Evaluate model  
7. Save trained model  



## ⚙️ Training Configuration

Training was performed using:

- **GPU acceleration (Google Colab)**
- **Batch size:** 4  
- **Epochs:** 3  
- **Maximum token length:** 128  
- **Evaluation strategy:** Per epoch  

These settings were optimized for stable training on Google Colab’s free GPU tier.



##  Evaluation

Model performance is evaluated using:

- Validation Loss  
- Accuracy  
- (Optional) F1 Score  

eval_loss: 0.21
eval_accuracy: 0.93
##  Key Concepts Demonstrated

- **Transfer Learning**  
  Leveraging a pretrained Transformer model and adapting it to a resume classification task.

- **Transformer-based NLP**  
  Utilizing DistilBERT for contextual text understanding and semantic representation.

- **Multi-class Classification**  
  Handling multiple job categories with proper label encoding and model configuration.

- **Label Encoding**  
  Converting categorical job roles into numerical labels for supervised learning.

- **Dataset Preprocessing**  
  Cleaning, splitting, and tokenizing textual resume data for model training.

- **Model Serialization**  
  Saving and loading trained models and encoders for reuse without retraining.

- **GPU Training in Google Colab**  
  Accelerated fine-tuning using GPU resources for faster and stable training.



##  Future Improvements

- Add confusion matrix visualization  
- Add F1-score and full classification report  
- Perform hyperparameter tuning  
- Implement resume PDF parsing pipeline  
- Build web interface using Streamlit  
- Develop REST API using Flask or FastAPI  
- Deploy trained model on Hugging Face Hub  



##  Why This Project Matters

This project demonstrates real-world AI engineering practices:

- Efficient use of pretrained Transformer models  
- Production-ready saving and loading workflow  
- Scalable NLP classification pipeline  
- Proper handling of multi-class classification problems  
- GPU-accelerated fine-tuning for performance optimization  
