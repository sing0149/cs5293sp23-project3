smartcity-prediction
Name: Sagar Singh
## Requirements
os
PdfReader (from PyPDF2)
re
contractions_dict (from contractions)
TfidfVectorizer (from sklearn.feature_extraction.text)
KMeans (from sklearn.cluster)
pickle
numpy
pandas
argparse

## Installations
If the above requirements are not already installed, they can be installed using pip or conda
This code requires the following libraries:

!pip install PyPDF2
!pip install contractions
!pip install scikit-learn
!pip install numpy
!pip install pandas

## Running the project
Clone the repository to the local machine or download the file in ZIP format and extract it.

The project can be run from the command line using the following command:
python smartcity-prediction.py --document path_to_document.pdf

This will predict the cluster for the new smart city applicant document and append the result to the output file.

## Workflow
The code is a Python script that predicts the cluster for a new smart city applicant document. It uses the K-Means clustering algorithm to cluster the documents based on their text content. The code uses the following workflow:

Importing the necessary libraries such as os, PdfReader (from PyPDF2), re, contractions_dict (from contractions), TfidfVectorizer (from sklearn.feature_extraction.text), KMeans (from sklearn.cluster), pickle, numpy, pandas, and argparse.

## Defining the following functions:

normalize_text(): This function applies text normalization to convert the text to lowercase, remove special characters, and reduce multiple whitespaces to a single whitespace.
expand_contractions(): This function expands contractions in the text using a contraction map.
remove_terms(): This function removes terms that may affect clustering and topic modeling.
process_pdf_file(): This function processes a PDF file, extracts the text, applies text normalization and contraction expansion, and removes terms.
process_document(): This function processes a document separately based on its file extension (PDF in this case).
load_pdf_data(): This function loads data from PDF files in a directory by calling process_pdf_file() for each file.
normalize_data_text(): This function normalizes the text in the data by calling normalize_text() for each document.
train_clustering_model(): This function trains the clustering model by vectorizing the text data using TfidfVectorizer and applying K-Means clustering.
save_model(): This function saves the trained clustering model to a file using pickle.
load_model(): This function loads a saved clustering model from a file using pickle.
predict_cluster(): This function predicts the cluster for a document by normalizing and removing terms from the text, transforming it using TfidfVectorizer, and applying the trained K-Means model.
append_to_output_file(): This function appends the predicted cluster information to the output file.

The main part of the script parses the command-line arguments using the argparse module. The script takes one argument, i.e., the path to the new smart city applicant document.

The script then loads data from PDF files in the specified directory using load_pdf_data(). It applies text normalization to the loaded data using normalize_data_text() and trains the clustering model using train_clustering_model().

If a new document is provided through the command-line argument, the script processes the document separately using process_document() and predicts the cluster for the document using predict_cluster(). The predicted cluster information is then appended to the output file using append_to_output_file().

Finally, the script prints a success message indicating that the output has been appended to the output file.

In summary, the script processes PDF files in a directory, normalizes the text data, trains a clustering model using K-Means, and predicts the cluster for a new smart city applicant document.

## test functions
Project3 Module Tests
This module contains tests for the functions in the project3 module.

Prerequisites
Python 3.x
pytest package (can be installed using pip install pytest)
Getting Started
Clone the repository or download the project3.py and test_project3.py files.

Install the required packages by running the following command:

Copy code
pip install -r requirements.txt
Ensure that the project3.py file is in the same directory as the test_project3.py file.

Running the Tests
To run the tests, use the following command:

Copy code
pytest test_project3.py
The tests will be executed, and the results will be displayed in the console.

Test Functions
test_normalize_text()
This test verifies the functionality of the normalize_text() function.
It checks if the text is correctly normalized to lowercase.
test_expand_contractions()
This test verifies the functionality of the expand_contractions() function.
It checks if contractions in the text are correctly expanded.
test_remove_terms()
This test verifies the functionality of the remove_terms() function.
It checks if specific terms are correctly removed from the text.
test_process_pdf_file()
This test verifies the functionality of the process_pdf_file() function.
It checks if a PDF file can be processed correctly and returns non-empty clean text.
test_process_document()
This test verifies the functionality of the process_document() function.
It checks if a document can be processed correctly and added to the data list.
test_load_pdf_data()
This test verifies the functionality of the load_pdf_data() function.
It checks if PDF files in a directory can be loaded correctly and added to the data list.
test_normalize_data_text()
This test verifies the functionality of the normalize_data_text() function.
It checks if the text in the data list is correctly normalized.
test_train_clustering_model()
This test verifies the functionality of the train_clustering_model() function.
It checks if a clustering model can be trained correctly using the provided data.
test_save_and_load_model()
This test verifies the functionality of the save_model() and load_model() functions.
It checks if a model can be saved to a file and loaded back correctly.
test_predict_cluster()
This test verifies the functionality of the predict_cluster() function.
It checks if a cluster can be correctly predicted for a test document using a trained model.
Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.

License
This project is licensed under the MIT License.