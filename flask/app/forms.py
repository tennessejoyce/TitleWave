from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class TextboxForm(FlaskForm):
    title = StringField('Post title', validators=[DataRequired()])
    submit = SubmitField('Evaluate title!')

class TextboxFormBig(FlaskForm):
    title = StringField('Question body', validators=[DataRequired()])
    submit = SubmitField('Suggest a title!')