import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


##########################################################
# Title: Phishing Detection Neural Network
# Model Type: Multilayer Neural Network
# Dataset: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning

# This dataset contains 48 features extracted from 5000 phishing webpages and 5000 legitimate webpages, 
# which were downloaded from January to May 2015 and from May to June 2017. 
# An improved feature extraction technique is employed by leveraging the browser automation framework (i.e., Selenium WebDriver), 
# which is more precise and robust compared to the parsing approach based on regular expressions.

# Anti-phishing researchers and experts may find this dataset useful for phishing features analysis, 
# conducting rapid proof of concept experiments or benchmarking phishing classification models.

# Goal: Classify URLs as phishing or legitimate.

# Main Features of the Dataset:
#     URL Length: The total length of the URL.
#     Number of Dots: The count of dots in the URL.
#     Subdomain Level: The depth of subdomains present in the URL.
#     Presence of Special Characters: Counts of special characters like @, _, -, etc.
#     Domain Age: The age of the domain in days.
#     HTTPS Availability: Indicates whether the URL uses HTTPS (a secure protocol).
#     IP Address: Whether the URL contains an IP address.
#     Malicious Content: Features indicating the presence of malware or suspicious content.

##########################################################


X_columns = ["id", "NumDots", "SubdomainLevel", "PathLevel", "UrlLength", "NumDash", 
    "NumDashInHostname", "AtSymbol", "TildeSymbol", "NumUnderscore", 
    "NumPercent", "NumQueryComponents", "NumAmpersand", "NumHash", 
    "NumNumericChars", "NoHttps", "RandomString", "IpAddress", 
    "DomainInSubdomains", "DomainInPaths", "HttpsInHostname", 
    "HostnameLength", "PathLength", "QueryLength", "DoubleSlashInPath", 
    "NumSensitiveWords", "EmbeddedBrandName", "PctExtHyperlinks", 
    "PctExtResourceUrls", "ExtFavicon", "InsecureForms", 
    "RelativeFormAction", "ExtFormAction", "AbnormalFormAction", 
    "PctNullSelfRedirectHyperlinks", "FrequentDomainNameMismatch", 
    "FakeLinkInStatusBar", "RightClickDisabled", "PopUpWindow", 
    "SubmitInfoToEmail", "IframeOrFrame", "MissingTitle", 
    "ImagesOnlyInForm", "SubdomainLevelRT", "UrlLengthRT", 
    "PctExtResourceUrlsRT", "AbnormalExtFormActionR", 
    "ExtMetaScriptLinkRT", "PctExtNullSelfRedirectHyperlinksRT"
]

file_path = 'data/phishing.csv'
df = pd.read_csv(file_path)

# PREPROCESSING
train_ratio = 0.99

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(len(df) * train_ratio)

 # 90% training, 10% testing
train_df = df.iloc[:split_index] 
test_df = df.iloc[split_index:]  

test_df.to_csv("data/test.csv", index=False)

print("Initial size of Dataframe: ",df.shape[1])

X = train_df[X_columns].to_numpy()
X_test = test_df[X_columns].to_numpy()

print("Initial size of Training set: ", X.shape[1])
print("Initial size of Testing set: ", X_test.shape[1])

Y = train_df["CLASS_LABEL"].to_numpy()

print("Sizes of X: ", X.shape, " and Y: ", Y.shape)

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,))   
print(Xt.shape, Yt.shape) 

# MODEL FITING AND TESTING
tf.random.set_seed(1234) 
model = Sequential(
    [
        tf.keras.Input(shape=(49,)),
        Dense(128, activation='sigmoid'),                                                                 
        Dense(64, activation='sigmoid'),                    
        Dense(32, activation='sigmoid'),                       
        Dense(16, activation='sigmoid'), 
        Dense(8, activation='sigmoid'),
        Dense(1, activation='sigmoid')                      
     ]
)

model.summary()


model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,         
    epochs=5,
)

# TESTING
def predict(X_test):
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    yhat = (predictions >= 0.5).astype(int)
    print(predictions)
    if yhat >= 0.5:
        print("Phishing.")
    else:
        print("Not Phishing.")

print(f"Testing {len(X_test)} cases...")
for i in range(len(X_test)):
    predict(X_test[i])

# Check the test csv file to make sure that the predictions are correct
