from flask import Flask, render_template, request
#from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoConfig
#import torch.nn.functional as F
#import torch
import pymysql
import os

app = Flask(__name__)

# Database connection setup
def get_db_connection():
    return pymysql.connect(
        host=os.getenv('RDS_HOST'), 
        user=os.getenv('RDS_USER'), 
        passwd=os.getenv('RDS_PASSWORD'),
        db=os.getenv('RDS_DB')
    )

import requests

API_URL = "https://hw6z6nsvw90cqbn7.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": os.getenv('HF_AUTHTOKEN'),
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


# Store data to database
def store_to_db(user_id, troubling_scenario, before_text, after_text, score_before, score_after):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = """
        INSERT INTO sentiment_analysis (user_id, troubling_scenario, before_text, after_text, score_before, score_after)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (user_id, troubling_scenario, before_text, after_text, score_before, score_after))
    conn.commit()
    cursor.close()
    conn.close()

#model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSequenceClassification.from_pretrained(model_name)


# sentiment scoring logic
def get_adjusted_score(sentiment_scores):
    pos_score, neg_score, neu_score = 0, 0, 0

    for result in sentiment_scores:
        if result['label'].lower() == 'positive':
            pos_score = result['score']
        elif result['label'].lower() == 'negative':
            neg_score = 1 - result['score']
        else:
            neu_score = result['score']

    if neu_score > max(pos_score, neg_score):
        if pos_score > neg_score:
            return 'positive', pos_score
        else:
            return 'negative', neg_score

    if pos_score > neg_score:
        return 'positive', pos_score
    elif neg_score > pos_score:
        return 'negative', neg_score
    else:
        return 'positive', pos_score


# sentiment label logic
def predict_sentiment(text):
    #input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    #with torch.no_grad():
    #    outputs = model(input_ids=input_ids)
    #    logits = outputs.logits[0]
    #    prob = F.softmax(logits, dim=-1).numpy()
        
    sent = query({
        "inputs": text,
        "parameters": {
                "top_k": 3
        }
    })
    
    prob_dict = {d["label"]: d["score"] for d in sent}
    
    pos_score = prob_dict['positive']
    neg_score = prob_dict['negative']

    adjusted_score = pos_score - neg_score

    if adjusted_score < -0.25:
        label = "Strongly Negative"
    elif -0.25 <= adjusted_score <= 0:
        label = "Leaning Negative"
    elif 0 < adjusted_score <= 0.25:
        label = "Leaning Positive"
    else:
        label = "Strongly Positive"

    return label, adjusted_score


# Starting page
@app.route('/')
def home():
    return render_template('home.html')


# Public use page - no A number required
@app.route('/public', methods=['GET', 'POST'])
def public_page():
    troubling_scenario = request.form.get('scenario', '')
    before_text = request.form.get('before_text', '')
    after_text = request.form.get('after_text', '')

    if request.method == 'POST':
        emotion_before, score_before = predict_sentiment(before_text)
        emotion_after, score_after = predict_sentiment(after_text)
        


        return render_template('public.html',
                               troubling_scenario=troubling_scenario,
                               before_text=before_text,
                               after_text=after_text,
                               emotion_before=emotion_before,
                               emotion_after=emotion_after,
                               score_before=round(score_before, 2),
                               score_after=round(score_after, 2))
    return render_template('public.html')


# USU use page - requires A number
@app.route('/usu', methods=['GET', 'POST'])
def usu_page():
    user_id = request.form.get('user_id', '')
    troubling_scenario = request.form.get('scenario', '')
    before_text = request.form.get('before_text', '')
    after_text = request.form.get('after_text', '')

    if request.method == 'POST':
        if not user_id:
            return render_template('usu.html',
                                   warning="Please enter your A number.",
                                   user_id=user_id,
                                   troubling_scenario=troubling_scenario,
                                   before_text=before_text,
                                   after_text=after_text)

        emotion_before, score_before = predict_sentiment(before_text)
        emotion_after, score_after = predict_sentiment(after_text)
        
        # Store the data in the RDS database
        store_to_db(user_id, troubling_scenario, before_text, after_text, round(score_before, 2), round(score_after, 2))

        return render_template('usu.html',
                               user_id=user_id,
                               troubling_scenario=troubling_scenario,
                               before_text=before_text,
                               after_text=after_text,
                               emotion_before=emotion_before,
                               emotion_after=emotion_after,
                               score_before=round(score_before, 2),
                               score_after=round(score_after, 2))

    return render_template('usu.html')


#if __name__ == '__main__':
#    app.run(debug=True)

