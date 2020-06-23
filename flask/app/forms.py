from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class TextboxForm(FlaskForm):
	'''Textbox to enter the proposed title for the webapp version.'''
    title = StringField('Post title', validators=[DataRequired()])
    submit = SubmitField('Evaluate title!')

class TextboxFormBig(FlaskForm):
	'''Textbox to enter the question body for the webapp version.'''
    title = StringField('Question body', validators=[DataRequired()])
    submit = SubmitField('Suggest a title!')