import pickle


# Load the classifier   
with open('C:/Users/Chief/Desktop/tn-police/ML/model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the vectorizer
with open('C:/Users/Chief/Desktop/tn-police/ML/count_vectorizer.pkl', 'rb') as f:
     vectorizer = pickle.load(f)


# Check if the classifier and vectorizer are working correctly
example_emails = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
example_counts = vectorizer.transform(example_emails)
predictions = clf.predict(example_counts)
print(predictions)



