from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('dataset.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/schedule')
def schedule():
    return render_template('schedule.html')

@app.route('/testing')
def testing():
    return render_template('testing.html')


if __name__ == '__main__':
    app.run()
