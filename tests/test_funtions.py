from project3 import normalize_text, process_pdf_file, remove_terms, expand_contractions, process_document, data, contractions_dict, load_pdf_data, train_clustering_model, normalize_data_text, TfidfVectorizer, KMeans, save_model, load_model, append_to_output_file,predict_cluster
import os


def test_normalize_text():
    text = "This is a Test Text!"
    normalized_text = normalize_text(text)
    assert normalized_text == "this is a test text"


def test_expand_contractions():
    text = "I can't do it. I won't be there."
    expanded_text = expand_contractions(text, contractions_dict)
    assert expanded_text == "I cannot do it. I will not be there."


def test_remove_terms():
    text = "The  is a  in the  of California."
    removed_text = remove_terms(text)
    assert removed_text == "The  is a  in the  of California."


def test_process_pdf_file():
    file_path = "smartcity/WIMadison.pdf"
    clean_text = process_pdf_file(file_path)
    assert clean_text is not None


def test_process_document():
    document_path = "smartcity/WIMadison.pdf"
    process_document(document_path)
    assert len(data) == 1


def test_load_pdf_data():
    directory = "smartcity/"
    load_pdf_data(directory)
    assert len(data) > 0


def test_normalize_data_text():
    test_data = [
        {"text": "This is Text 1."},
        {"text": "This is Text 2."}
    ]
    normalize_data_text(test_data)
    assert test_data[0]["text"] == "this is text 1"
    assert test_data[1]["text"] == "this is text 2"


def test_train_clustering_model():
    test_data = [
        {"text": "This is Text 1."},
        {"text": "This is Text 2."}
    ]
    vectorizer, model = train_clustering_model(test_data, num_clusters=2)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(model, KMeans)


def test_save_and_load_model():
    test_model = KMeans(n_clusters=2, random_state=42)
    model_file = "test_model.pkl"
    save_model(test_model, model_file)
    loaded_model = load_model(model_file)
    assert isinstance(loaded_model, KMeans)




def test_predict_cluster():
    test_document_text = "This is a test document."
    vectorizer = TfidfVectorizer()
    model = KMeans(n_clusters=2, random_state=42)
    train_data = [
        {"text": "This is Text 1."},
        {"text": "This is Text 2."}
    ]
    normalize_data_text(train_data)
    X_train = [item['text'] for item in train_data]
    vectorizer.fit(X_train)
    model.fit(vectorizer.transform(X_train))
    predicted_cluster = predict_cluster(test_document_text, vectorizer, model)
    assert predicted_cluster in [0, 1]





