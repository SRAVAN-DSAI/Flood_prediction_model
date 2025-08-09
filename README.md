\# Flood Prediction Web App 🌊



!\[Project Status](https://img.shields.io/badge/status-active-brightgreen)



An interactive web application that predicts the probability of flooding based on various environmental, infrastructural, and socio-economic factors. This project uses a highly accurate LightGBM machine learning model trained on a comprehensive dataset.



!\[Streamlit App Screenshot](https://raw.githubusercontent.com/SRAVAN-DSAI/Flood\_prediction\_model/main/flood\_prediction\_app\_screenshot.png)



---



\## 🚀 Features



\-   \*\*Interactive UI\*\*: Adjust 20 different real-world factors using sliders to see their immediate impact on flood risk.

\-   \*\*Accurate Predictions\*\*: Powered by a fine-tuned LightGBM model with an R-squared score of over 93%.

\-   \*\*Feature Engineering\*\*: The model uses engineered features like `LandslideRisk` and `InadequateInfrastructure` for improved accuracy.

\-   \*\*Easy to Use\*\*: Simple, intuitive interface built with Streamlit.

\-   \*\*Deployable\*\*: Ready for deployment on platforms like Streamlit Cloud.



---



\## 🛠️ Tech Stack



\-   \*\*Language\*\*: Python

\-   \*\*Machine Learning\*\*: Scikit-learn, LightGBM

\-   \*\*Web Framework\*\*: Streamlit

\-   \*\*Data Handling\*\*: Pandas, Joblib

\-   \*\*Version Control\*\*: Git \& Git LFS (for handling large model files)



---



\## 📁 Project Structure



```



Flood\\\_prediction\\\_model/

├── .gitattributes          \\# Configures Git LFS to track large files

├── .python-version         \\# Specifies the Python version for deployment

├── app.py                  \\# The main Streamlit application script

├── lgbm\\\_flood\\\_prediction\\\_model.joblib  \\# The saved, pre-trained model

├── requirements.txt        \\# List of Python dependencies

└── README.md               \\# This file



````



---



\## ⚙️ Setup and Installation



To run this project locally, follow these steps:



1\.  \*\*Clone the Repository\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/SRAVAN-DSAI/Flood\_prediction\_model.git](https://github.com/SRAVAN-DSAI/Flood\_prediction\_model.git)

&nbsp;   cd Flood\_prediction\_model

&nbsp;   ```



2\.  \*\*Set up Git LFS\*\*

&nbsp;   This project uses Git Large File Storage (LFS) to manage the large model file.

&nbsp;   ```bash

&nbsp;   # Install Git LFS (if you haven't already)

&nbsp;   git lfs install



&nbsp;   # Pull the large model file from LFS storage

&nbsp;   git lfs pull

&nbsp;   ```



3\.  \*\*Create a Virtual Environment\*\* (Recommended)

&nbsp;   ```bash

&nbsp;   python -m venv venv

&nbsp;   source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`

&nbsp;   ```



4\.  \*\*Install Dependencies\*\*

&nbsp;   Install all the required Python libraries from the `requirements.txt` file.

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



---



\## ▶️ How to Run the App



Once the setup is complete, you can run the Streamlit application with a single command:



```bash

streamlit run app.py

````



Your web browser will automatically open a new tab with the running application. You can now interact with the sliders in the sidebar to get real-time flood probability predictions.



-----



\## 🧠 Model Details



The prediction model is a \*\*LightGBM Regressor\*\*. The final version was trained on a dataset with 18 carefully engineered features.



&nbsp; - \*\*Feature Engineering\*\*: Two new features (`LandslideRisk` and `InadequateInfrastructure`) were created by combining four of the original features. The original four were then dropped to avoid data redundancy.

&nbsp; - \*\*Performance\*\*: The model achieved a final \*\*R-squared score of \\~0.94\*\*, indicating it can explain about 94% of the variability in flood probability, making it a highly reliable predictor.



-----



\## ☁️ Deployment



This application is designed for easy deployment on Streamlit Cloud. The necessary configuration files (`requirements.txt`, `.python-version`) are included in the repository.



-----



\## 🤝 Contributing



Contributions are welcome\\! If you have ideas for improvements or find any issues, please feel free to open an issue or submit a pull request.



\## 📄 License



This project is licensed under the MIT License. See the `LICENSE` file for more details.

