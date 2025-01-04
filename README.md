# Skin Disease Classification with SVM

This project classifies skin diseases into nine categories using Support Vector Machines (SVM) and VGG16 for feature extraction.

## Class Labels
1. Actinic keratosis
2. Atopic Dermatitis
3. Benign keratosis
4. Dermatofibroma
5. Melanocytic nevus
6. Melanoma
7. Squamous cell carcinoma
8. Tinea Ringworm Candidiasis
9. Vascular lesion

## Steps to Run
1. Place your training and testing datasets in the `data/` directory.
2. Install dependencies: `pip install -r requirements.txt`.
3. Train the SVM model using `notebooks/svm_training.ipynb`.
4. Run the Flask app: `python app/app.py`.
5. Access the app at `http://127.0.0.1:5000`.
