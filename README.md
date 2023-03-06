# Customer Segmentation with RFM using [FLO](https://www.flo.com.tr/)'s Dataset

![RFM Segments](https://raw.githubusercontent.com/ArmanTunga/rfm-flo/6159dd4debeb1a9b1d0295337d3ebb34963df2ea/src/rfm-segments.png)

This is the RFM _Real Life Case Study_ of [Miuul](https://miuul.com/) Data Science & Machine Learning Bootcamp

## About FLO

---
Flo is a big Turkish fashion retail company, which has been operating since 2010. It offers various
products
such as shoes, bags, and clothes for both men and women. Flo has an extensive network of stores across Turkey and has
recently expanded its operations to other countries.

## Business Problem

---

FLO wants to segment its customers and determine marketing strategies according to these segments.

For this purpose, the behaviors of the customers will be defined and groups will be formed according to these behavior
clusters.

## Story of Dataset

---

The dataset consists of information obtained from the past shopping behaviors of
customers who made their last purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.

- master_id: Unique customer number

- order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)

- last_order_channel : The channel where the most recent purchase was made

- first_order_date : Date of the customer's first purchase

- last_order_date : Customer's last purchase date

- last_order_date_online : The date of the last purchase made by the customer on the online platform

- last_order_date_offline : Last shopping date made by the customer on the offline platform

- order_num_total_ever_online : The total number of purchases made by the customer on the online platform

- order_num_total_ever_offline : Total number of purchases made by the customer offline

- customer_value_total_ever_offline : Total fee paid by the customer for offline purchases

- customer_value_total_ever_online : The total fee paid by the customer for their online shopping

- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

---

## TASKS

### TASK 1: Data Understanding and Preparation

1. Read the flo_data_20K.csv data.

2. In the dataset:
    1. The first 10 observations,
    2. Variable names,
    3. Descriptive statistics,
    4. Null value,
    5. Variable types, review.

3. Omnichannel means that customers shop from both online and offline platforms. Create new variables for each
   customer's
   total purchases and spend.
4. Examine the variable types. Change the type of variables that express date to date.
5. See the breakdown of the number of customers, average number of products purchased, and average spend across shopping
   channels.
6. Rank the top 10 customers with the highest revenue.
7. List the top 10 customers with the most orders.
8. Streamline the data preparation process.

### TASK 2: Calculating RFM Metrics

### TASK 3: Calculating RF and RFM Scores

### TASK 4: Segment Definition of RF Scores

### TASK 5: Time for action!

1. Examine the recency, frequency and monetary averages of the segments.

2. With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer IDs to
   the csv.

    1. FLO includes a new women's shoe brand. The product prices of the brand it includes are above the general customer
       preferences.
       For this reason, customers in the profile who will be interested in the promotion of the brand and product sales
       are requested to be contacted privately.
       Those who shop from their loyal customers (champions, loyal_customers), on average over 250 TL and from the women
       category, are the customers to contact privately.
       Save the id numbers of these customers in the csv file as new_brand_target_customer_id.cvs.

    2. Up to 40% discount is planned for Men's and Children's products.
       It is aimed to specifically target customers who are good customers in the past, but who have not shopped for a
       long time,
       who are interested in the categories related to this discount, who should not be lost, who are asleep and new
       customers.
       Save the ids of the customers in the appropriate profile to the csv file as discount_target_customer_ids.csv.

### TASK 6: Functioning the whole process.

Write a function that does whole the process
