# Master-Thesis
## Overcoming Institutional Data Isolation: A Benchmark Study on Federated Learning for Credit Scoring

### Abstract
Recent years saw a huge influx of more and more potent machine learning models using delicate methods that involved an increasing amount of
complexity. Business entities in their strive to find an edge over their competitors observed this development with delight and incorporated
machine and deep learning methods into their business model. But even the most potent models can only recover a limited amount of information
from a given set of data, which is why new ideas are needed to broaden the spectrum of information a single business can tap into. Such an idea
is federated learning which promises a knowledge transfer without ever revealing sensitive information. Although institutional federated
learning gained more momentum recently, the majority of empirical studies in this area still focuses on classical deep learning
applications like image processing on a multitude of edge devices instead of a small group of business entities. However, to enforce a
widespread adoption of federated learning across many different industries, it needs comprehensive benchmark studies that are
specifically tailored towards one of these industries. A good example for one of said industries is credit scoring and retail banking, due to
its financial resources and rich data environment. To close the mentioned research gap, this thesis therefore aims at giving a thorough
statistical evaluation of the predictive capabilities of federated learning for this industry. This is supposed to help practitioners
making the decision for adopting federated learning in order to promote a more productive, decentral and privacy preserving machine learning
framework.

Key words: Federated Learning, Credit Scoring, Homomorphic Encryption

### Data 
Data can be retrieved here: https://www.kaggle.com/c/home-credit-default-risk
If you want to run the experiments, please make sure to download the data files from the above URL into the "raw_data" folder (currently empty).

### config_general
Contains a few high level parameters that are used throughout multiple script files. Only change if absolutely necessary.

### Instructions for code execution
For the reproduction of the results, first install a new virtual environment with Python 3.8.8 (no Anaconda/Conda distribution!) then 
install the necessary dependencies from the requirements.txt file. This guide was written to work on Windows 10, no warranties for Mac or Linux.

#### Steps:
```
1.  Open a terminal and navigate to the location of the project folder.

2.  Now set up a virtual environment using venv
    -   py -m pip install --user virtualenv		# installs venv 
    -   py -m venv env					# creates virtual environment

3.  Activate the environment:
    -   cd env\Scripts
    -   activate

4.  Go back to the main project directory

5.  Ensure that your Python version is 3.8.8 
    -   py --version

6.  Install dependencies from requirements.txt (I used pip version 20.2.3)
    -   pip install -r requirements.txt

7.  Execute the main script which will go through the code to reproduce the experiments itself. Total runtime was approx. 30h on my machine.
    -   py main.py
```

