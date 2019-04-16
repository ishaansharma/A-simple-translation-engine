import flask
import os
import fast_inference
import random

app = flask.Flask(__name__, template_folder="templates")
runner_one, runner_two = fast_inference.get_runners()

@app.route('/', methods=['GET'])
def webpage():
    return flask.render_template('index.html')

@app.route('/translate_tahitian', methods=['POST'])
def translate_tahitian():
    content = flask.request.get_json(silent=True)
    tahitian_input = content['tahitian_input']
    print('tahitian INPUT = %s' %tahitian_input)
    i_file = 'txt_files/%s.txt' %random.randint(0,10**20)
    o_file = 'txt_files/%s.txt' %random.randint(0,10**20)
    open(i_file, 'w').write(tahitian_input)
    runner_one.infer(
    features_file = i_file,
    predictions_file=o_file,
    checkpoint_path='run/avg'
    )
    english_output = open(o_file).read()
    print('ENGLISH OUTPUT = %s' %english_output)
    return flask.jsonify({'english_output': english_output})

@app.route('/translate_english', methods=['POST'])
def translate_english():
    content = flask.request.get_json(silent=True)
    english_input = content['english_input']
    print('ENGLISH INPUT %s' %english_input)
    i_file = 'txt_files/%s.txt' %random.randint(0,10**20)
    o_file = 'txt_files/%s.txt' %random.randint(0,10**20)
    open(i_file, 'w').write(english_input)
    runner_two.infer(
    features_file = i_file,
    predictions_file=o_file,
    checkpoint_path='run_rev/avg'
    )
    tahitian_output = open(o_file).read()
    print('tahitian OUTPUT = %s' %tahitian_output)
    return flask.jsonify({'tahitian_output': tahitian_output})

app.run('0.0.0.0', 8000)
