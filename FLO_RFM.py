###############################################################
# Customer Segmentation with RFM
###############################################################
import datetime

import pandas as pd

###############################################################
# Business Problem
###############################################################
# FLO wants to segment its customers and determine marketing strategies according to these segments.
# For this purpose, the behaviors of the customers will be defined and groups will be formed according to these behavior clusters.

###############################################################
# Story of Dataset
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of
# customers who made their last purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Customer's last purchase date
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Last shopping date made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
# customer_value_total_ever_online : The total fee paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

###############################################################
# TASKS
###############################################################

# TASK 1: Data Understanding and Preparation
# 1. Read the flo_data_20K.csv data.
# 2. In the dataset:
# a. The first 10 observations,
# b. Variable names,
# c. Descriptive statistics,
# d. Null value,
# e. Variable types, review.
# 3. Omnichannel means that customers shop from both online and offline platforms. Create new variables for each customer's total purchases and spend.
# 4. Examine the variable types. Change the type of variables that express date to date.
# 5. See the breakdown of the number of customers, average number of products purchased, and average spend across shopping channels.
# 6. Rank the top 10 customers with the highest revenue.
# 7. List the top 10 customers with the most orders.
# 8. Streamline the data preparation process.

# TASK 2: Calculating RFM Metrics

# TASK 3: Calculating RF and RFM Scores

# TASK 4: Segment Definition of RF Scores

# TASK 5: Time for action!
# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
# 2. With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer IDs to the csv.
# a. FLO includes a new women's shoe brand. The product prices of the brand it includes are above the general customer preferences.
# For this reason, customers in the profile who will be interested in the promotion of the brand and product sales are requested to be contacted privately.
# Those who shop from their loyal customers (champions, loyal_customers), on average over 250 TL and from the women category, are the customers to contact privately.
# Save the id numbers of these customers in the csv file as new_brand_target_customer_id.cvs.

# b. Up to 40% discount is planned for Men's and Children's products.
# It is aimed to specifically target customers who are good customers in the past, but who have not shopped for a long time,
# who are interested in the categories related to this discount, who should not be lost, who are asleep and new customers.
# Save the ids of the customers in the appropriate profile to the csv file as discount_target_customer_ids.csv.


# TASK 6: Functioning the whole process.

###############################################################
# TASK 1: Data Understanding and Preparing
###############################################################
import datetime as dt
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)

df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
# 2. In the dataset:
# a. The first 10 observations,
df.head(10)
# b. Variable names,
df.columns
# c. Shape,
df.shape
# d. Descriptive statistics,
df.describe().T
# e. Null value,
df.isnull().sum()
# f. Variable types, review.
df.info()  # df.dtypes gives only types

# 3. Omnichannel means that customers shop from both online and offline platforms. Create new variables for each customer's total purchases and spend.
df["order_num_total"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 4. Examine the variable types. Change the type of variables that express date to date.
df.dtypes
df.head(2)
date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].astype("datetime64[ns]")
df.dtypes
# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime) another option but I try to avoid using apply


# 5. See the distribution of the number of customers in the shopping channels, the total number of products purchased and the total expenditures.
(
    df
    .groupby("order_channel")
    .agg({"master_id": "count",
          "order_num_total": "sum",
          "customer_value_total": "sum"})
)

# 6. Rank the top 10 customers with the highest revenue.
"""
This is only to see the customer_value_total better. To do so we change the order of the first variable(master_id) and the variable customer_value_total

cols = list(df.columns)
cols[0], cols[df.columns.get_loc("customer_value_total")] = cols[df.columns.get_loc("customer_value_total")], cols[0]
df = df[cols]
"""
df.sort_values(by="customer_value_total", ascending=False).head(10).reset_index()

# 7. List the top 10 customers with the most orders.
df.sort_values(by="order_num_total", ascending=False).head(10).reset_index()


# 8. Functioning the data preparation process.
def prepare_data(dataframe: pd.DataFrame):
    """
    This func prepares data of dataframe by calculating total purchases and spend money for each customers and
    assigning new variables.

    Args:
        dataframe: The dataframe that we want to prepare data from

    Returns: dataframe: Data prepared dataframe

    """
    # Creating new variables for each customer's total purchases and spend.
    dataframe["order_num_total"] = (
            dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]).astype(int)
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]
    # Change the type of variables that express date to date.
    date_columns = [col for col in dataframe.columns if "date" in col]
    dataframe[date_columns] = dataframe[date_columns].astype("datetime64[ns]")
    return dataframe


df = prepare_data(df)

###############################################################
# TASK 2: Calculating RFM Metrics
###############################################################

# Analysis date shall be 2 days after the last shopping date in the data set
last_order_date = df["last_order_date"].max()
last_order_date = dt.datetime.strptime(str(last_order_date), "%Y-%m-%d %H:%M:%S")
today_date = last_order_date + dt.timedelta(days=2)

# A new rfm dataframe with customer_id, recency, frequnecy and monetary values
df.head(1)  # only to see column names and what they refer to

rfm = df.groupby("master_id").agg({"last_order_date": lambda date: (today_date - date.iloc[0]).days,
                                   "order_num_total": lambda num: num,
                                   "customer_value_total": lambda value: value,
                                   })

rfm = rfm.reset_index()
rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
###############################################################
# TASK 3: Calculating RF and RFM Scores
###############################################################

#  Converting Recency, Frequency and Monetary metrics to scores between 1-5
#  with the help of qcut and recording these scores as recency_score, frequency_score and monetary_score
rfm["recency_score"] = pd.qcut(x=rfm["recency"], q=5, labels=["5", "4", "3", "2", "1"])
rfm["frequency_score"] = pd.qcut(x=rfm["frequency"].rank(method="first"), q=5, labels=["1", "2", "3", "4", "5"])
rfm["monetary_score"] = pd.qcut(x=rfm["monetary"], q=5, labels=["1", "2", "3", "4", "5"])

# Expressing recency_score and frequency_score as a single variable and saving it as RF_SCORE
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
rfm.head()
###############################################################
# TASK 4: Segment Definition of RF Scores
###############################################################

# Segment definition and converting RF_SCORE to segments with the help of defined seg_map so that the generated RFM scores can be explained more clearly.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

###############################################################
# TASK 5: Time for action!
###############################################################
rfm.dtypes
# 1. Examine the recency, frequency and monetary averages of the segments.
rfm.groupby("segment")["recency", "frequency", "monetary"].mean()
# rfm.groupby("segment")["recency", "frequency", "monetary"].agg(["mean", "count"])

# 2. With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer IDs to the csv.

# a. FLO includes a new women's shoe brand. The product prices of the brand it includes are above the general customer preferences.
# For this reason, customers in the profile who will be interested in the promotion of the brand and product sales are requested to be contacted privately.
# These customers were planned to be loyal(loyal and champions) and female shoppers who's monetary is bigger than 250. Save the id numbers of the customers to the csv file as new_brand_target_customer_id.cvs.
rfm.head()
rfm["segment"].unique()  # to see how we named loyal customers and champions.
loyal_customers_champions = rfm[(rfm["segment"].isin(["loyal_customers", "champions"])) & (rfm["monetary"] > 250)][
    "customer_id"]

loyal_customers_champions.head()
type(loyal_customers_champions)  # pandas.core.series.Series
df.head()
df["master_id"].isin(loyal_customers_champions)
df["interested_in_categories_12"].head(25)
# df[~df["interested_in_categories_12"].str.contains("ERKEK") & df["interested_in_categories_12"].str.contains("KADIN")]
df[df["interested_in_categories_12"].str.contains("KADIN")].head()

new_brand_target_customer_id = (
    df  # From our first df
    .loc[df["master_id"].isin(loyal_customers_champions) &  # Find loyal customers(loyal customers and champions) AND
         df["interested_in_categories_12"].str.contains("KADIN")]  # Find
    ["master_id"]
)
new_brand_target_customer_id.name = "customer_id"  # Change name attribute from master_id to customer_id
new_brand_target_customer_id.to_csv("datasets/new_brand_target_customer_id.csv")  # save to csv

# b. Up to 40% discount is planned for Men's and Children's products.
# We want to specifically target customers who are good customers in the past
# who are interested in categories related to this discount, but have not shopped for a long time and new customers.
# Save the ids of the customers in the appropriate profile to the csv file as discount_target_customer_ids.csv.
target_customers = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]
discount_target_customer_ids = (
    df
    .loc[df["master_id"].isin(target_customers) &
         df["interested_in_categories_12"].str.contains("ERKEK|COCUK")
         ]
)
discount_target_customer_ids.name = "customer_id"
discount_target_customer_ids.to_csv("datasets/discount_target_customer_ids.csv")


###############################################################
# TASK 6: Functioning the whole process.
###############################################################
# 8. Functioning the data preparation process.
def prepare_data(dataframe: pd.DataFrame):
    """
    This func prepares data of dataframe by calculating total purchases and spend money for each customers and
    assigning new variables.

    Args:
        dataframe: The dataframe that we want to prepare data from

    Returns: dataframe: Data prepared dataframe

    """
    # Creating new variables for each customer's total purchases and spend.
    dataframe["order_num_total"] = (
            dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]).astype(int)
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]
    # 4. Examine the variable types. Change the type of variables that express date to date.
    date_columns = [col for col in dataframe.columns if "date" in col]
    dataframe[date_columns] = dataframe[date_columns].astype("datetime64[ns]")
    return dataframe


def create_rfm(dataframe: pd.DataFrame, csv: bool = False, file_name: str = "rfm") -> pd.DataFrame:
    """
    This function creates an RFM dataframe with segments extracted with some calculations, from the given dataframe.

    Args:
        dataframe: Dataframe that we want to create segments from (unprepared)
        csv: Should I save as a csv too? If true then save.
        file_name: What should be the file name when I save the csv? Default is "rfm".

    Returns:
        rfm: RFM dataframe containing segment, recency, frequency, monetary information as variables and score values of this variables too.
    """

    dataframe = prepare_data(dataframe)

    # Calculating RFM Metrics
    last_order_date = dataframe["last_order_date"].max()
    last_order_date = dt.datetime.strptime(str(last_order_date), "%Y-%m-%d %H:%M:%S")
    today_date = last_order_date + dt.timedelta(days=2)
    rfm = dataframe.groupby("master_id").agg({"last_order_date": lambda date: (today_date - date.iloc[0]).days,
                                              "order_num_total": lambda num: num,
                                              "customer_value_total": lambda value: value,
                                              })
    rfm = rfm.reset_index()
    rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
    # Calculating RF and RFM Scores

    #  Converting Recency, Frequency and Monetary metrics to scores between 1-5
    rfm["recency_score"] = pd.qcut(x=rfm["recency"], q=5, labels=["5", "4", "3", "2", "1"])
    rfm["frequency_score"] = pd.qcut(x=rfm["frequency"].rank(method="first"), q=5, labels=["1", "2", "3", "4", "5"])
    rfm["monetary_score"] = pd.qcut(x=rfm["monetary"].rank(method="first"), q=5, labels=["1", "2", "3", "4", "5"])

    # Expressing recency_score and frequency_score as a single variable and saving it as RF_SCORE
    rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    # Segment definition and converting RF_SCORE to segments with the help of defined seg_map so that the generated RFM scores can be explained more clearly.
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

    if csv:
        rfm.to_csv(f"{file_name}.csv")
    return rfm


rfm = create_rfm(df)
