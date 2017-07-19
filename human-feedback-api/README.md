
## Usage

```
python manage.py collectstatic
python manage.py migrate
gunicorn -b 0.0.0.0:8000 human_feedback_site.wsgi --log-file -
```

## Acknowledgments 
Some code originally derived from the [heroku python starter kit](https://github.com/heroku/python-getting-started).
