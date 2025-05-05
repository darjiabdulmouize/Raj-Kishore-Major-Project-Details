import streamlit as st
import base64
import sqlite3
import subprocess

# ================ Database Connection Function ===============
def create_connection():
    return sqlite3.connect("dbs.db")

# ================ Background Image Function ===============
def add_bg_from_local(image_file):
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
add_bg_from_local('5.jpeg')

# ================ Navigation Handling ===============
if "page" not in st.session_state:
    st.session_state.page = "home"

def set_page(new_page):
    st.session_state.page = new_page
    st.rerun()  # Updated Fix

# ================ Navbar ===============

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Home"):
        set_page("home")
with col2:
    if st.button("Register"):
        set_page("reg")
with col3:
    if st.button("Login"):
        set_page("login")

# ================ Home Page ===============
if st.session_state.page == "home":
    st.markdown('<h1 style="color:#000;text-align: center;font-size:50px;font-family:Caveat, sans-serif;">'
                'Mining the Opinions of Software Developers for Improved Project Insights</h1>', 
                unsafe_allow_html=True)

    description = (
        "This project analyzes software developers' opinions using transfer learning, "
        "text pre-processing, feature extraction, and machine learning classification to gain insights."
    )
    st.markdown(f'<h2 style="color:#000;text-align: justify;font-size:20px;font-family:Caveat, sans-serif;">{description}</h2>', 
                unsafe_allow_html=True)

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

    name = st.text_input("Enter your name", key="reg_name")
    password = st.text_input("Enter your password", type="password", key="reg_password")
    confirm_password = st.text_input("Confirm your password", type="password", key="reg_confirm_password")
    email = st.text_input("Enter your email", key="reg_email")
    phone = st.text_input("Enter your phone number", key="reg_phone")

    if st.button("REGISTER", key="register_button"):
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

# ================ Login Page ===============
elif st.session_state.page == "login":
    st.markdown("<h1 style='text-align: center;'>Login Here</h1>", unsafe_allow_html=True)

    conn = create_connection()
    name = st.text_input("User name", key="login_name")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
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

    conn.close()

# ================ Default Page (Error Handling) ===============
else:
    st.error("Page not found. Please use the main app.")
