import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


spam_df = pd.read_csv("spam.csv")
#print(spam_df.head(5))

spam_df.groupby('Category').describe()

#turning the category column spam/ham into a new column called spam with numerical variables, 1 for spam, 0 for not spam
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

#create train-test-split

x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25)

#print(x_train.describe())

# find the word count and store data as matrix 

cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
x_train_count.toarray()


#train mode 

model = MultinomialNB()
model.fit(x_train_count, y_train)

#pre-test ham; ham is real email
email_ham = ['hey would you like to go to the movies with me?']
email_ham_count = cv.transform(email_ham)  # this code transforms to numbers as above

model.predict(email_ham_count)

#pre test spam email; should return 1 
email_spam = ['Login verification']
email_spam_count = cv.transform(email_spam)
model.predict(email_spam_count)

#test model 

x_test_count = cv.transform(x_test)
print(model.score(x_test_count, y_test))

