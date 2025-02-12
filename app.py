from flask import Flask,request,render_template
import pickle
import pandas as pd

app= Flask(__name__)

model=pickle.load(open('trained_model.pkl', 'rb'))
transformer=pickle.load(open('transformer.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form
    age = int(data['age'])
    sex = data['sex']
    bmi = float(data['bmi'])
    children = int(data['children'])
    smoker = data['smoker']
    region = data['region']
    
    test_df=pd.DataFrame({
      'age':[age],
      'sex':[sex],
      'bmi':[bmi],
      'children':[children],
      'smoker':[smoker],
      'region':[region]
      })
    
    test_df_transformed=transformer.transform(test_df)
    
    prediction = model.predict(test_df_transformed)[0]
    
    return render_template('result.html', prediction=prediction.round(2))


if __name__ == '__main__':
    app.run(debug=True)
    
    
