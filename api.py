import csv
import pickle
from fastapi import FastAPI, Body, Request, status
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from urlextract import URLExtract
from googletrans import Translator
app = FastAPI()
extractor = URLExtract()


@app.get("/api/v1/list")
def List():
   print("Hello")
   return FileResponse("req.txt", media_type='text/csv', filename="UUID_List.csv")

@app.post("/api/v1/sms_recive")
async def receive_sms(request: Request):
    # Parse the incoming SMS message from the request
    data = await request.json()
    message = data.get('message')
    sender = data.get('sender')
    
    with open('list/UUID.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if sender in row:
                urls = extractor.find_urls(message)
                print(urls)
                return {'list check': 'failed', 'url': urls}
    with open('D:/App/api/ML/model.pkl', 'rb') as f:
        clf = pickle.load(f)

# Load the vectorizer
    with open('C:/Users/imsan/Projects/Git/api/ML/count_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    translator=Translator()
    translated_text=translator.translate(message)
    example=translated_text.text
    l=[]
    l.append(example)

# Check if the classifier and vectorizer are working correctly
    example_counts = vectorizer.transform(l)
    predictions = clf.predict(example_counts)
    print(predictions)
    if predictions==1:
        urls = extractor.find_urls(message)
        print(urls)
        return {'List Check': 'success','ML Check':'spam', 'url': urls}
    else:
        return {'List Check': 'success','ML Check':'ham'}
        




#message will be sent to ml if found return spam or if not found ham
#sender will be sent to check list first if found return responce spam if now found ham

# @app.post("/api/v1/sms_receive")
# async def receive_sms(request: Request):
    
#     data = await request.json()
#     message = data.get('message')

#     // Load the TensorFlow.js library
#     const tf = require('@tensorflow/tfjs');

#     // Load the converted model
#     const model = await tf.loadLayersModel('path/to/model.json');

#     // Prepare input data for prediction
#     const input = tf.tensor2d([[message]]);

#     // Make prediction
#     const output = model.predict(input);

#     // Return the prediction as a response
#     prediction = output.dataSync();
#     return {'status': 'success', 'prediction': prediction};



