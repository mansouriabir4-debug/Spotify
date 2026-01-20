import streamlit as st
import requests
from flask import jsonify
from dotenv import load_dotenv
import os

load_dotenv()

def fetch_data():
    with st.spinner("Working..."):
        ### prepare API key and URL
        api_url = os.getenv("BACKEND_URL") + f"/"

        ### invoke API and get the response
        response = requests.get(url=api_url)
        if response.status_code == requests.codes.ok:
            ### Convert data to JSON format
            data = response.json()
            print(data)
            status = "OK"
        else:
            data = {"code": response.status_code, "message": response.text}
            status = "ERROR"
    return status, data