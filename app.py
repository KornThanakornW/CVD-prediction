import streamlit as st
from multiapp import MultiApp
from apps import data, prediction, information

app = MultiApp()


app.add_app("Prediction", prediction.app)
app.add_app("Information", information.app)
app.add_app("Data", data.app)

app.run()



