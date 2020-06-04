from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class TextboxForm(FlaskForm):
    title = StringField('Post title', validators=[DataRequired()])
    submit = SubmitField('Predict!')