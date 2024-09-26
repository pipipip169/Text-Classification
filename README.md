# Spam Text Classification using Naive Bayes

This project uses Natural Language Processing (NLP) techniques and a Gaussian Naive Bayes classifier to identify spam messages. It processes text messages and classifies them as either spam or not spam (ham).

## Features

- **Spam Detection**: Classify text messages as either spam or not spam (ham) using a Naive Bayes model.
- **NLP Preprocessing**: The text is preprocessed with lowercasing, punctuation removal, tokenization, stopword removal, and stemming.
- **Evaluation Metrics**: Accuracy of the model is measured on validation and test datasets.
- **Simple Prediction API**: Easily predict whether a message is spam by calling the `predict` function.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- nltk

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/pipipip169/Text-Classification.git
    cd Text-Classification
    ```

2. **Install requirements**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK resources**:

    Make sure you download the required NLTK resources by running the following commands in your Python environment:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

## Usage

1. **Prepare the dataset**:

    Place your dataset in the root directory of the project. The dataset should be in CSV format with at least the following columns:
    - **Message**: The text message to be classified.
    - **Category**: The label (either 'spam' or 'ham').

    For this example, the dataset file is named `2cls_spam_text_cls.csv`.

2. **Run the model**:

    You can run the `Spam_Text_Classification_Naive_Bayes.ipynb` notebook to train the model and evaluate its performance. The notebook walks you through all the steps:

    - Data preprocessing
    - Training the Naive Bayes model
    - Model evaluation
    - Making predictions on new text

3. **Prediction Example**:

    You can use the `predict` function to classify new text messages:

    ```python
    test_input = 'I am actually thinking a way of doing something useful'
    prediction_cls = predict(test_input, model, dictionary)
    print(f'Prediction: {prediction_cls}')
    ```

## Project Structure

- `Spam_Text_Classification_Naive_Bayes.ipynb`: The main notebook containing the project implementation.
- `2cls_spam_text_cls.csv`: The dataset containing messages and their corresponding labels (not included, you will need to supply your own).
- `README.md`: This file.
- `requirements.txt`: A list of the Python libraries required for the project.

## Contributing

If you would like to contribute, feel free to submit issues or pull requests to improve the project. Contributions in improving the model, adding features, or optimizing the code are welcome.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- The dataset used for this project is inspired by various spam detection datasets available online.
- The project uses NLTK for natural language processing and scikit-learn for machine learning.
