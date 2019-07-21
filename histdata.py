from flask import redirect, request, Flask, render_template, url_for
import json, requests
import pandas as pd
import joblib
from datetime import datetime
from yahoo_historical import Fetcher
import dataset_predict

app=Flask(__name__)

@app.route('/menu')
def menuhistdata():
    return render_template('menuhist.html')

# @app.route('/histdata',methods=['POST','GET'])
# def histdata():
#     namasaham=request.form['namasaham']
#     tgl1=request.form['tanggal1']
#     tgl2=request.form['tanggal2']

if __name__=='__main__':
    app.run(debug=True)