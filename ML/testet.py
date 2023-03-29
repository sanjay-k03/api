import pickle
from googletrans import Translator

# Load the classifier   
with open('C:/Users/Chief/Desktop/tn-police/ML/model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the vectorizer
with open('C:/Users/Chief/Desktop/tn-police/ML/count_vectorizer.pkl', 'rb') as f:
     vectorizer = pickle.load(f)

translator = Translator()
translated_text = translator.translate('मैं आज जल्द ही घर आऊंगा, मैं आज रात इन चीजों के बारे में बात नहीं करना चाहता, मैं बहुत रो चुका हूं।')
example = translated_text.text
l=[]
l.append(example)
# Check if the classifier and vectorizer are working correctly
example_emails = [l]
#example_emails = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
example_counts = vectorizer.transform(l)
predictions = clf.predict(example_counts)
print(predictions)



