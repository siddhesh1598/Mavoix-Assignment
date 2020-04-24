from flask import Flask, jsonify, request, render_template, make_response
from main import getAnswer

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
	return render_template("index.html")

@app.route('/output', methods = ['POST'])
def output():
	req = request.get_json()
	# print("Image URL: ", req['url'])
	text = getAnswer(req['url'])
	res = make_response(jsonify({"output": text}))

	# return render_template("output.html", text_to_display=text)
	return res
	    
if __name__ == "__main__":
    app.run(debug=True)