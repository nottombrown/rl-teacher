from flask import Flask, render_template

app = Flask(__name__, static_folder='/tmp/rl_teacher_media', static_url_path='')

@app.route('/')
def test():
    return 'flask server running'

if __name__ == '__main__':
    app.run()