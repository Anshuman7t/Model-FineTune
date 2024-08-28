# Model-FineTune For Chat and Text Generation

#Guide: Fine-Tuning a Language Model Using Custom Data

Before starting, ensure you have the following installed:

* Python (version 3.8 or later)
* pandas, datasets, transformers, torch, accelerate, bitsandbytes and peft libraries
* Access to a GPU for faster training (recommended) (Also you can use Google Colab for all this)

#Data Preparation

Data should be in a structured format, such as a CSV or Excel file, containing questions and answers.

Step 1: Load and Clean Data
#code:
import pandas as pd

# Load the Excel file
file_path = '/content/FAQ_list_PAN_TAN_All.xlsx'
df = pd.read_excel(file_path)

# Initialize an empty list to store the cleaned data
cleaned_data = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    questions = row['Question'].split('\n')  # Split multiple questions in one cell
    answer = row['Answer'].strip()  # Get the corresponding answer

    for question in questions:
        question = question.strip()  # Clean up any extra whitespace
        if question:  # Ensure the question is not empty
            cleaned_data.append({'Question': question, 'Answer': answer})

# Convert the cleaned data to a DataFrame and save it
cleaned_df = pd.DataFrame(cleaned_data)
cleaned_df.to_csv('./cleaned_dataset.csv', index=False)


