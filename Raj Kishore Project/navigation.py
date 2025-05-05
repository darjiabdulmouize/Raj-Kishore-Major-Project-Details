import streamlit as st
import base64
import sqlite3
import re
import subprocess

# ================ Database Connection Function ===============
def create_connection():
    """Creates a connection to the SQLite database."""
    return sqlite3.connect("dbs.db")

# ================ Background Image Function ===============
def add_bg_from_local(image_file):
    """Adds a background image from a local file."""
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Error: {image_file} not found. Please check the path.")

# Set background image
add_bg_from_local('1.png')

# ================ Navigation Handling ===============
if "page" not in st.session_state:
    st.session_state.page = "home"  # Default to home page

def set_page(new_page):
    """Updates the session state for navigation."""
    st.session_state.page = new_page

# Debugging output
st.write("Current Page:", st.session_state.page)

# ================ Home Page ===============
if st.session_state.page == "home":
    st.markdown('<h1 style="color:#000;text-align: center;font-size:50px;font-family:Caveat, sans-serif;">'
                'Mining the Opinions of Software Developers for Improved Project Insights</h1>', 
                unsafe_allow_html=True)

    st.text(" ")
    description = (
        "This project aims to analyze software developers' opinions using transfer learning, leveraging techniques such as "
        "text pre-processing, feature extraction, and machine learning classification to gain insights."
    )
    st.markdown(f'<h2 style="color:#000;text-align: justify;font-size:20px;font-family:Caveat, sans-serif;">{description}</h2>', 
                unsafe_allow_html=True)

    if st.button("Go to Registration"):
        set_page("reg")

    if st.button("Go to Login"):
        set_page("login")

# ================ Registration Page ===============
elif st.session_state.page == "reg":
    st.markdown("<h1 style='text-align: center;'>Register Here</h1>", unsafe_allow_html=True)

    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY, 
                        name TEXT NOT NULL, 
                        password TEXT NOT NULL, 
                        email TEXT NOT NULL UNIQUE, 
                        phone TEXT NOT NULL)''')

    name = st.text_input("Enter your name")
    password = st.text_input("Enter your password", type="password")
    confirm_password = st.text_input("Confirm your password", type="password")
    email = st.text_input("Enter your email")
    phone = st.text_input("Enter your phone number")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("REGISTER"):
            if password == confirm_password:
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE email=?", (email,))
                if cur.fetchone():
                    st.error("User with this email already exists!")
                else:
                    cur.execute("INSERT INTO users (name, password, email, phone) VALUES (?, ?, ?, ?)", 
                                (name, password, email, phone))
                    conn.commit()
                    st.success("User registered successfully!")
                    set_page("login")
            else:
                st.error("Passwords do not match!")

    conn.close()

    with col2:
        if st.button("Go to Login"):
            set_page("login")

# ================ Login Page ===============
elif st.session_state.page == "login":
    st.markdown("<h1 style='text-align: center;'>Login Here</h1>", unsafe_allow_html=True)

    conn = create_connection()
    name = st.text_input("User name")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
            user = cur.fetchone()
            if user:
                st.success(f"Welcome back, {user[1]}! Login successful!")
                st.write("Redirecting to Background.py...")
                subprocess.Popen(["streamlit", "run", "Background.py"])
                st.stop()
            else:
                st.error("Invalid user name or password!")
    
    with col2:
        if st.button("Go to Registration"):
            set_page("reg")

    conn.close()

# ================ Default Page (Error Handling) ===============
else:
    st.error("Page not found. Please use the main app.")
