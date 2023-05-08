from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import numpy as np
import pandas as pd
import argparse
import regex
import re
from PyPDF2 import PdfReader
import os
pdf_dir = 'smartcity/'
output_file = 'smartcity_predict.tsv'
data = []
all_text = []
cities_to_remove = []
contractions_dict = {
    "can't": 'cannot',
    "won't": 'will not',
    "I'm": 'I am',
    "he's": 'he is',
    "she's": 'she is',
    "it's": 'it is',
    "that's": 'that is',
    "there's": 'there is',
    "we're": 'we are',
    "they're": 'they are',
    # Add more contractions as needed
}

def normalize_text(text):
    # Apply text normalization
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def expand_contractions(text, contraction_map):
    # Function to expand contractions in the text
    contraction_pattern = re.compile('({})'.format('|'.join(contraction_map.keys())),
                                     flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_map.get(match)
            if contraction_map.get(match)
            else contraction_map.get(match.lower())
        )
        if expanded_contraction:
            expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contraction_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_terms(text):
    # Function to remove terms that may affect clustering and topic modeling
    terms_to_remove = ['city', 'state', 'smart', 'page']
    for term in terms_to_remove:
        text = text.replace(term, "")
    return text


def process_pdf_file(file_path):
    # Process a PDF file and extract the text
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        raw_text = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            raw_text += page_text
            all_text.append(page_text)

        clean_text = expand_contractions(raw_text, contractions_dict)
        clean_text = normalize_text(clean_text)

        if clean_text.strip() == "":
            return None
        else:
            clean_text = remove_terms(clean_text)
            return clean_text


def process_document(document_path):
    # Process the new document separately
    if os.path.isfile(document_path) and document_path.endswith('.pdf'):
        city_name = os.path.splitext(os.path.basename(document_path))[0]
        clean_text = process_pdf_file(document_path)

        if clean_text is None:
            print(f"The document '{document_path}' could not be processed correctly.")
        else:
            data.append({
                'city': city_name,
                'text': clean_text
            })


def load_pdf_data(directory):
    # Load data from PDF files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            city_name = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            clean_text = process_pdf_file(file_path)

            if clean_text is not None:
                data.append({
                    'city': city_name,
                    'text': clean_text
                })
            else:
                cities_to_remove.append(city_name)


def normalize_data_text(data):
    # Normalize the text in the data
    for d in data:
        d['text'] = normalize_text(d['text'])


def train_clustering_model(data, num_clusters=5):
    # Train the clustering model
    text_data = [d['text'] for d in data]
    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(text_data)

    # Train the clustering model
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(vectorized_data)

    return vectorizer, model


def save_model(model, model_file):
    # Save the model to a file
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_file):
    # Load the model from a file
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_cluster(document_text, vectorizer, model):
    # Predict the cluster for a document
    normalized_text = normalize_text(document_text)
    clean_text = remove_terms(normalized_text)
    document_vector = vectorizer.transform([clean_text]).toarray()
    predicted_cluster = model.predict(document_vector)[0]
    return predicted_cluster


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the cluster for a new smart city applicant')
    parser.add_argument('--document', type=str, help='Path to the new smart city applicant document')
    args = parser.parse_args()

    # Load data from PDF files in the directory
    load_pdf_data(pdf_dir)

    # Process the new document separately
    if args.document and args.document.endswith('.pdf'):
        process_document(args.document)

    # Remove cities that were not processed correctly
    data = [d for d in data if d['city'] not in cities_to_remove]

    # Normalize the text data
    normalize_data_text(data)

    # Train the clustering model
    vectorizer, model = train_clustering_model(data, num_clusters=5)

    # Save the model
    model_file = 'model.pkl'
    save_model(model, model_file)

    # Predict the cluster for the new document
    if data:
        new_document_text = data[-1]['text']
        predicted_cluster = predict_cluster(new_document_text, vectorizer, model)

        # Print the output
        print(f"[{data[-1]['city']}] clusterid: {predicted_cluster}")
        print("Summary:", new_document_text[:50] + "...")
        print("Keywords:", ", ".join(data[-1]['text'].split()[:5]))

        # Append the new city to the output file
        output_data = pd.DataFrame(data, columns=['city', 'text'])
        output_data['clean_text'] = output_data['text'].apply(normalize_text)
        output_data['clusterid'] = predicted_cluster

        if os.path.isfile(output_file):
            output_data.to_csv(output_file, sep='\t', index=False, mode='a', header=False)
        else:
            output_data.to_csv(output_file, sep='\t', index=False)

        print(f"Output appended to {output_file} successfully.")
    else:
        print("No new document provided or no processed cities found.")

