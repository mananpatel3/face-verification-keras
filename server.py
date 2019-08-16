from flask import Flask, render_template, request, url_for, redirect
import numpy as np


app = Flask(__name__)
from flask_face_verification_main import single_image
from PIL import Image
from io import BytesIO


@app.route('/', methods=['GET', 'POST'])
def home():
    global compare
    if request.method == 'GET':
        return render_template('home.html')

    elif request.method == 'POST':

        a = request.files['file1']
        b = request.files['file2']

        if a.filename != '' and b.filename != '':
            img_a = BytesIO()
            img_b = BytesIO()

            a.save(img_a)
            b.save(img_b)
            img_1 = np.array(Image.open(img_a))
            img_2 = np.array(Image.open(img_b))

            compare = single_image(img_1, img_2)
            if (compare == "Authentication Successful"):
                return redirect(url_for('success'))
            else:
                return redirect(url_for('fail'))


        else:
            return redirect(request.url)
    else:
        return redirect(request.url)




@app.route('/success', methods=['GET', 'POST'])
def success():
    return render_template('success.html', result=compare)

@app.route('/fail', methods=['GET', 'POST'])
def fail():
    return render_template('fail.html', result=compare)




if __name__ == '__main__':
    app.run()

