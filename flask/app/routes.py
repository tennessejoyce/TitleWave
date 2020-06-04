from flask import render_template, flash, redirect, request, jsonify
from app import app
from app.forms import TextboxForm
from app.bert_quality import predict_quality
from app.summarize import suggest_a_title
import sys

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = TextboxForm()
    if form.validate_on_submit():
        title = form.title.data
        prob = predict_quality(title)
        flash(f'Post title: {title}')
        flash(f'Probability of getting an answer: {prob}')
    return render_template('index.html', title='Enter the proposed title of your Stack Overflow post', form=form)

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    print('Evaluating title...')
    title = request.get_json()['title']
    prob = predict_quality(title)
    return jsonify(prob)

@app.route('/suggest', methods=['GET', 'POST'])
def suggest():
    print('Suggesting title...')
    body = request.get_json()['body']
    print(body, file=sys.stderr)
    title = suggest_a_title(body)
    return jsonify(title)