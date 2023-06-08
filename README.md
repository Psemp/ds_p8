# Deploying a model with AWS EMR
- [Project information](https://openclassrooms.com/fr/paths/164/projects/633/assignment)
- [Data : Fruits 360](https://www.kaggle.com/moltean/fruits)
- Using PySpark with EMR

# Goals and Steps :
- Deploying a pretrained model on an AWS EMR Cluster with PCA on features
- Making available the data and the results (see links)
- Optimal `k_components` for PCA determined via Sklearn's PCA on a subset of data, aiming for 95% explained variance on local subset (see `notebook_local_execution.ipynb`)
- Saving as both parquet (efficient) and csv (easy to share as one file) (see links)
- Location of servers in GDPR enforcing countries : Ireland is used both for storage (s3) and Cluster hosting (although other EU hosts can have sometimes lower "spot" purchase options)
- Cluster is m5.xlarge (1 Master and 2 Slaves), m5 instances are good for multipurpose but the slave nodes can be changed either to c5 or r5 instances depending on spot prices (more optimized towards calculations)

# Links :
- [Notebook executed and hosted on S3](https://ds-p8.s3.eu-west-1.amazonaws.com/jupyter/jovyan/nb_aws_spark.ipynb)
- [Image dataset in .zip format (a script and csv can also be used to download all images from S3)](https://ds-p8.s3.eu-west-1.amazonaws.com/images_upload.zip)
- [Results as csv (initially parquet but export as csv for simplicity)](https://ds-p8.s3.eu-west-1.amazonaws.com/csvs/results_fruits.csv) (also available in this repo, with deduced html links to each image)
