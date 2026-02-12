# Medical NLP chatbot 

This projects uses NLP with intent matching to create a chatbot for users to access medical records. The chatbot can be interacted with using a GUI created with PyQt5.

Tests to find which type of ML model would be most accurate at intent matching can be found in: 
    - Coursework 
        - code
            - model_expriments.ipynb 

A file containing the training of support vector machines for intent machine can be found in: 
    - Coursework
        - code 
            - model_trainer_and_tests

The main code for this project can be found in: 
    - Coursework
        - code 
            - main.py
    


# Comments 

If you wish to try this Marvin out please do. Find some user details from data/processed_health_data.csv and try it out. 
You'll see that the dataset uses ages and not dates of birth, so provided you supply a date of birth that matches the age of the person you've choosen you won't run into any issues.

I use - bobby jackson - 11 april 1995 

accessable medical records - Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results