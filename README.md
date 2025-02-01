# Diabetes Prediction Risks

![Shutterstock Image](https://www.uhhc.co.nz/wp-content/uploads/2018/12/shutterstock_571889917.jpg)

## Description

The purpose of predicting diabetes risk using machine learning is to identify individuals at high risk of developing diabetes by analyzing health metrics such as glucose levels, BMI, and age. Machine learning models can detect patterns and correlations in large datasets that may not be apparent through traditional methods, enabling early diagnosis and intervention. This approach helps prevent severe complications like heart disease and kidney failure by promoting timely lifestyle changes and medical care. Additionally, it reduces healthcare costs by focusing resources on at-risk individuals and improving overall public health outcomes. Ultimately, predicting diabetes risk with machine learning aims to enhance quality of life through proactive and personalized healthcare solutions.

## Motivation

Diabetes is a global health concern, and early detection can significantly improve patient outcomes. This project aims to:

1.Identify at-risk individuals early to prevent complications like heart disease and kidney failure.

2.Promote proactive healthcare by encouraging lifestyle changes and timely medical intervention.

3.Leverage machine learning to analyze health data and provide accurate predictions.

4.Demonstrate the use of FastAPI for building scalable and efficient APIs.

5.Serve as a learning tool for developers interested in machine learning and API development.

## Deployment

The Diabetes Prediction Risk is deployed using FastAPI and can be accessed locally or hosted on a cloud platform. Below are the steps to deploy and use the API:

## Prerequisites

1. Install Python 3.7+.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Steps to Run the API

## Train the Model:

```bash
Run train_model.py
```

## python train_model.py

This will create the app/models/ folder containing diabetes_rf_model.pkl and scaler.pkl.

## Start the FastAPI Server:

```bash
uvicorn app.main:app --reload
```

Open https://ml-predecting-diabetes-4.onrender.com in your browser..

## Folder Structure

```bash
diabetes_api/
├── app/
│ ├── **init**.py
│ ├── main.py
│ ├── models/
│ │ ├── diabetes_rf_model.pkl
│ │ └── scaler.pkl
│ └── schemas.py
├── train_model.py
├── requirements.txt
├── README.md
└── tests/
└── test_api.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or feedback, please contact [Samuel esubalew] at [esubalew.samiye@gmail.com].

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Samimt1/ML-Predecting-Diabetes.git
   cd diabetes_api
   ```
