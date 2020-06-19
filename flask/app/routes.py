from flask import render_template, flash, redirect, request, jsonify
from app import app
from app.forms import TextboxForm, TextboxFormBig
from app.nlp_models import suggest_title,evaluate_title
import sys

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form1 = TextboxForm()
    if form1.title.data and form1.validate_on_submit():
        title = form1.title.data
        prob = evaluate_title(title)
        predict_line = f'Probability of getting an answer: {prob}'
    else:
        predict_line = ''
    form2 = TextboxFormBig()
    if form2.title.data and form2.validate_on_submit():
        title = form2.title.data
        sugg = suggest_title(title)
        flash(f'Suggested title:')
        flash(sugg)
    return render_template('index.html', title='Enter the proposed title of your Stack Overflow post',
                             form1=form1, form2=form2, predict_line=predict_line)

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    print('Evaluating title...')
    title = request.get_json()['title']
    prob = evaluate_title(title)
    return jsonify(prob)

@app.route('/suggest', methods=['GET', 'POST'])
def suggest():
    print('Suggesting title...')
    body = request.get_json()['body']
    print(body, file=sys.stderr)
    title = suggest_title(body)
    return jsonify(title)

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

