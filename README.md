# Starbucks Rewards Project

### Repo Url

https://github.com/roysakura/starbucks-rewards


### Projects Overview

Starbucks app will send out some reward offers to customers. In this project, I will use dataset from Starbucks to analyze which demographic groups respond best to which offer type. And also, I will build a model to predict how good any people to any offer.

### Problem Statement

In this project, I want to find out the demogrphic model as defining customers' response to coupon. I will try to use complete rate as target for prediction. This is a Binary Classification problem. As the target is 1 or 0 after data processing. By using this model, we can predict what complete rate would be for any customer to any coupon type.  

### Files Structure

`data` folder contains all dataset for this project, and data preprocessing model process_data.py

`models` folder contains model building and tunning function train_classifier.py, it built the model and then save in the folder.

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
          
### Results

After fine tuning the xgboost classifier, the final result for predicting complete rate is 88.46% accuracy.

![](
https://news-material.oss-cn-shenzhen.aliyuncs.com/results.png)

By using tree algorithm, we can see importance of each feature. From the following graph. We can see that average transaction amount and loyaty are very important in coupon complete rate.

![](
https://news-material.oss-cn-shenzhen.aliyuncs.com/fetures_importance.png)

### Further Discussion

In this project, although we can predict the complete rate for customer in using coupon. We can't use this prediction to understand if a coupon can really increase customers' afterward transaction times. To further research, we need to find out which coupon or what kinds of coupon can increase or stimulus customers' transction actions.
 




