# app.py

from flask import Flask, render_template, request
from kmeans import run_kmeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load model once
df, model, scaler = run_kmeans()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        gender = request.form.get('gender')
        age = int(request.form.get('age'))
        income = int(request.form.get('income'))
        score = int(request.form.get('score'))

        gender_val = 0 if gender.lower() == 'male' else 1
        new_customer = [[gender_val, age, income, score]]
        scaled_customer = scaler.transform(new_customer)
        prediction = int(model.predict(scaled_customer)[0])

    # Plot clusters and save
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Income', y='Score', hue='Cluster', data=df, palette='Set2')
    plt.title('Customer Segmentation (K-Means)')
    plt.savefig('static/cluster_plot.png')
    plt.close()

    return render_template('index.html', image_path='static/cluster_plot.png', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
