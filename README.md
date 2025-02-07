# Slogan Quality Predictor 🚀

## Overview
The **Slogan Quality Predictor** is a machine learning model that evaluates the effectiveness of a slogan based on textual and numerical features. It helps businesses determine if their slogans are strong, unique, and marketable.

## Features
- **Text Processing**: Uses **TF-IDF Vectorization** to convert slogan text into numerical data.
- **Numerical Features**: Considers **business revenue, trademark conflicts, and social media availability**.
- **Prediction Model**: Uses **Ridge Regression** to predict the quality of a slogan.
- **User-Friendly Interface**: Built with **Streamlit** for easy slogan evaluation.

## Dataset
The dataset consists of the following features:
- `Slogan`: The text of the slogan.
- `Raw_Revenue_Info`: Estimated revenue of the business.
- `identical_trademarks`: Whether identical trademarks exist (binary: 0/1).
- `simillar_trademarks`: Whether similar trademarks exist (binary: 0/1).
- `social_media_taken`: Whether the social media handle is taken (binary: 0/1).
- `domains_point_to_website`: The number of domains pointing to the business website (target variable).

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/slogan-quality-predictor.git
   cd slogan-quality-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Model Training
The model is trained using **TF-IDF for text processing** and **StandardScaler for numerical feature scaling**. It uses a **Ridge Regression model** to predict the `domains_point_to_website` score, which indicates the strength of the slogan.

### Training Steps:
1. **Load Dataset**: Read `Slogans.csv` and preprocess missing values.
2. **Feature Extraction**:
   - Convert text into **TF-IDF vectors**.
   - Standardize numerical features.
3. **Train-Test Split**: Split data into **80% training, 20% testing**.
4. **Train Model**: Fit a **Ridge Regression** model.
5. **Evaluate Model**:
   - Mean Squared Error (MSE)
   - R-squared Score (R²)
6. **Save Model**: Export trained model, vectorizer, and scaler using `joblib`.

## Usage
To predict a slogan's quality, enter the slogan and related business details in the **Streamlit UI**. The app returns:
- A **prediction score** (higher = better slogan impact).
- A recommendation on whether the slogan is **strong or needs refinement**.

## Example Prediction
```python
slogan = "Just Do It"
raw_revenue = 80
identical_trademarks = 0
similar_trademarks = 1
social_media_taken = 1

prediction, label = predict_slogan(slogan, raw_revenue, identical_trademarks, similar_trademarks, social_media_taken)
print(f"Score: {prediction:.2f} - {label}")
```

## Technologies Used
- **Python** (pandas, numpy, sklearn, joblib)
- **Machine Learning** (TF-IDF, Ridge Regression, StandardScaler)
- **Streamlit** (for UI)

## Future Enhancements
- Incorporate **deep learning models** for more accurate predictions.
- Add more **business-specific metrics** to improve slogan quality analysis.
- Enhance UI with **visual insights** into slogan effectiveness.

## License
This project is licensed under the MIT License.

---

