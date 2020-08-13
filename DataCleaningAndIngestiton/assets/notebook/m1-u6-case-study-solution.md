## Case Study - Data ingestion

The goal of this case study is to put into practice the important takeaways from module 1.  We will go through the basic process that begins with refining the business opportunity and ensuring that it is articulated using a scientific thought process.

The business opportunity and case study was first mentioned in Unit 2 of module 1 and like the AAVIAL company itself these data were created for learning purposes.  We will be using the AAVAIL example as a basis for this case study.  [Watch the video again](https://vimeo.com/348708995) if you need a refresher.  You will be gathering data from several provided sources, staging it for quality assurance and saving it in a target destination that makes sense in this context.

Watch the video to review the important concepts from the units you just covered and to see an overview of the objectives for this case study.


```python
from IPython.display import IFrame    
IFrame('https://www.youtube.com/embed/NpEaa2P7qZI', width=600,height=400)
```





<iframe
    width="600"
    height="400"
    src="https://www.youtube.com/embed/NpEaa2P7qZI"
    frameborder="0"
    allowfullscreen
></iframe>




### Case study objectives

1. Given a business opportunity gather relevant data from multiple sources
2. Clean the gathered data
3. Create a script that gathers and cleans the data

### Getting started

Download this notebook and open it either locally using a Jupyter server or use your IBM cloud account to login to Watson Studio.  Inside of Waston Studio cloud if you have not already ensure that this notebook is loaded as part of the *project* for this course.

**You will need the following files to complete this case study**

* [m1-u6-case-study.ipynb](m1-u6-case-study.ipynb)
* [m1-u6-case-study-solution.ipynb](./notebooks/m1-u6-case-study-solution.ipynb)
* [aavail-customers.db](./data/aavail-customers.db)
* [aavail-steams.csv](./data/aavail-streams.csv)


1. Fill in all of the places in this notebook marked with ***YOUR CODE HERE*** or ***YOUR ANSWER HERE***
2. When you have finished the case study there will be a short quiz

You may review the rest of this content as part of the notebook, but once your are ready to get started be ensure that you are working with a *live* version either as part of Watson Studio or locally.

The data you will be sourcing from is contained in two sources.

1. A database ([SQLite](https://www.sqlite.org/index.html)) of `customer` data
2. A [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values) of `stream` level data

   >You will create a simple data pipeline that
   (1) simplifies the data for future analysis and 
   (2) performs quality assurance checks.

The process of building *the data ingestion pipeline* entails extracting data, transforming it, and loading into an appropriate data storage technology.  When constructing a pipeline it is important to keep in mind that they generally works in batches.  Data may be compiled during the day and the batch could be processed during the night.  The data pipeline may also be optimized to execute as a streaming computation that is, every event is handled as it occurs.

### PART 1: Gathering the data

The following is an [Entity Relationship Diagram (ERD)](https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model) that details the tables and contents of the database.

<img src="./images/aavail-schema.svg" alt="customer database schema" style="width: 600px;"/>


```python
## all the imports you will need for this case study
import os
import pandas as pd
import numpy as np
import sqlite3
```

Much of the data exists in a database.  You can connect to is using the `sqlite3` package with the following function.  Note that is is good practice to wrap your connect functions in a [try-except statement](https://docs.python.org/3/tutorial/errors.html) to cleanly handle exceptions.


```python
def connect_db(file_path):
    try:
        conn = sqlite3.connect(file_path)
        print("...successfully connected to db\n")
    except Error as e:
        print("...unsuccessful connection\n",e)
    
    return(conn)
```


```python
## make the connection to the database
conn = connect_db('./data/aavail-customers.db')

## print the table names
tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")]
print(tables)
```

    ...successfully connected to db
    
    ['CUSTOMERS', 'INVOICES', 'INVOICE_ITEMS']


#### QUESTION 1:

**extract the relevant data from the DB**

Query the database and extract the following data into a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).
 
* Customer ID (integer)
* Last name
* First name
* DOB
* City
* State
* Country (the name NOT the country_id)
* Gender

Remember that that SQL is case-insensitive, but it is traditional to use ALL CAPS for SQL keywords. It is also a convention to end SQL statements with a semi-colon.  

##### Resources

* [W3 schools SQL tutorial](https://www.w3schools.com/sql)
* [W3 schools SQL joins](https://www.w3schools.com/sql/sql_join.asp)


```python
## YOUR CODE HERE



```

#### QUESTION 2:

**Extract the relevant data from the CSV file**

For each ```customer_id``` determine if a customer has stopped their subscription or not and save it in a dictionary or another data container.


```python
df_streams = pd.read_csv(r"./data/aavail-streams.csv")
df_streams.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>stream_id</th>
      <th>date</th>
      <th>subscription_stopped</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1356</td>
      <td>2018-12-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1540</td>
      <td>2018-12-04</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1395</td>
      <td>2018-12-11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1255</td>
      <td>2018-12-22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1697</td>
      <td>2018-12-23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## YOUR CODE HERE


```

### PART 2: Cleaning the data

Sometimes it is known in advance which types of data integrity issues to expect, but other times it is during the Exploratory Data Analysis (EDA) process that these issues are identified.  After extracting data it is important to include checks for quality assurance even on the first pass through the AI workflow.  Here you will combine the data into a single structure and provide a couple checks for quality assurance.

#### QUESTION 3: 

**Implement checks for quality assurance**

1. Remove any repeat customers based on ```customer_id```
2. Remove stream data that do not have an associated ```stream_id```


```python
## YOUR CODE HERE



```

#### QUESTION 4: 

**combine the data into a single data structure**

For this example, the two most convenient structures for this task are Pandas dataframes and NumPy arrays.  At a minimum ensure that your structure accommodates the following.

1. A column for `customer_id`
2. A column for `country`
3. A column for ```age``` that is created from ```DOB```
4. A column ```customer_name``` that is created from ```first_name``` and ```last_name```
5. A column to indicate churn called ```is_subscriber```
7. A column that indicates ```subscriber_type``` that comes from ```invoice_item```
6. A column to indicate the total ```num_streams```

##### Resources

* [Python's datetime library](https://docs.python.org/3/library/datetime.html)
* [NumPy's datetime data type](https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html)



```python
## YOUR CODE HERE



```

### PART 3: Automating the process

To ensure that you code can be used to automate this process.  First you will save you dataframe or numpy array as a CSV file.  

#### QUESTION 5:

**Take the initial steps towards automation**

1. Save your cleaned, combined data as a CSV file.
2. From the code above create a function or class that performs all of the steps given a database file and a streams CSV file.
3. Run the function in batches and write a check to ensure you got the same result that you did in the code above.

There will be some logic involved to ensure that you do not write the same data twice to the target CSV file.

Shown below is some code that will split your streams file into two batches. 


```python
## code to split the streams csv into batches
data_dir = os.path.join(".","data")
df_all = pd.read_csv(os.path.join(data_dir,"aavail-streams.csv"))
half = int(round(df_all.shape[0] * 0.5))
df_part1 = df_all[:half]
df_part2 = df_all[half:]
df_part1.to_csv(os.path.join(data_dir,"aavail-streams-1.csv"),index=False)
df_part2.to_csv(os.path.join(data_dir,"aavail-streams-2.csv"),index=False)
```

You will need to save your function as a file.  The following cell demonstrates how to do this from within a notebook. 


```python
%%writefile aavail-data-ingestor.py
#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import sqlite3

data_dir = os.path.join(".","data")

pass
```

    Overwriting aavail-data-ingestor.py


You will also need to be able to pass the file names to your function without hardcoding them into the script itself.  This is an important step towards automation.  Here are the two libraries commonly used to accomplish this in Python.

* [getopt](https://docs.python.org/3/library/getopt.html)
* [argparse](https://docs.python.org/3/library/argparse.html)

You may run the script you just created from the commandline directly or from within this notebook using:


```python
!python aavail-data-ingestor.py aavail-customers.db aavail-streams-1.csv
```

Run the script once for each batch that you created and then load both the original and batch versions back into the notebook to check that they are the same. 


```python
## YOUR CODE HERE



```

#### QUESTION 6:

**How can you improve the process?**

In paragraph form or using bullets write down some of the ways that you could improve this pipeline.

YOUR ANSWER HERE





```python

```
