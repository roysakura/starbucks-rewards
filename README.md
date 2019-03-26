# Starbucks Rewards Project

### Repo Url

https://github.com/roysakura/starbucks-rewards


### Projects Description

Starbucks app will send out some reward offers to customers. In this project, I will use dataset from Starbucks to analyze which demographic groups respond best to which offer type. And also, I will build a model to predict how good any people to any offer.

### Instructions:
1. Run the following commands in the project's root directory to set up your evironments.

	  `python3 -m venv ./venv`

	- Activate virtual environment

		`source /venv/bin/activate`
	- Install dependant packages
	
	`pip install -r requirments.txt`
	
2. Load in the data and do cleaning and feature engineering.

    - To run data praparation
      
      `python data/process_data.py`
        
    - To run ML building and saving
      
      `python models/train_classifier.py`
          



