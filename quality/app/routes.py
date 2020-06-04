from flask import render_template, flash, redirect
from app import app
from app.forms import TextboxForm
from app.bert_quality import predict_quality

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = TextboxForm()
    if form.validate_on_submit():
        title = form.title.data
        prob = predict_quality(title)
        flash(f'Post title: {title}')
        flash(f'Probability of getting an answer: {prob}')
    return render_template('login.html', title='Enter the proposed title of your Stack Overflow post', form=form)
