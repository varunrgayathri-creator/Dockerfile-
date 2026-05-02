import gradio as gr
import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict(weight, sugar, area, product_type, mrp, year, size, city, store_type):
    data = np.array([
        weight, sugar, area, product_type, mrp,
        year, size, city, store_type
    ]).reshape(1, -1)

    prediction = model.predict(data)[0]
    return float(prediction)

gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Product Weight"),
        gr.Number(label="Sugar Content (encoded)"),
        gr.Number(label="Allocated Area"),
        gr.Number(label="Product Type (encoded)"),
        gr.Number(label="MRP"),
        gr.Number(label="Store Establishment Year"),
        gr.Number(label="Store Size (encoded)"),
        gr.Number(label="City Type (encoded)"),
        gr.Number(label="Store Type (encoded)")
    ],
    outputs=gr.Number(label="Predicted Sales"),
    title="📊 Sales Forecasting App"
).launch()