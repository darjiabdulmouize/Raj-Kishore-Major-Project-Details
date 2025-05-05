import streamlit as st
import base64
import sqlite3
import smtplib
import random

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
            input {{
                background-color: azure;
                border-radius: 5px;
                border: 1px solid black;
            }}
            input:hover {{
                background-color: lightblue;
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
if "otp" not in st.session_state:
    st.session_state.otp = None
if "reset_user" not in st.session_state:
    st.session_state.reset_user = None

def set_page(new_page):
    st.session_state.page = new_page
    st.rerun()

# ================ Navbar ===============
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Home", key="home_btn"):
        set_page("home")
with col2:
    if st.button("Register", key="register_btn"):
        set_page("reg")
with col3:
    if st.button("Login", key="login_btn_nav"):
        set_page("login")

# ================ Home Page ===============
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align: center;'>Mining the Opinions of Software Developers</h1>", unsafe_allow_html=True)

# ================ Registration Page ===============
elif st.session_state.page == "reg":
    st.markdown("<h1 style='text-align: center;'>Register Here</h1>", unsafe_allow_html=True)
    
    conn = create_connection()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY, 
                        name TEXT NOT NULL, 
                        password TEXT NOT NULL, 
                        email TEXT NOT NULL UNIQUE, 
                        phone TEXT NOT NULL)''')
    conn.commit()
    
    name = st.text_input("Enter your name")
    password = st.text_input("Enter your password", type="password")
    confirm_password = st.text_input("Confirm your password", type="password")
    email = st.text_input("Enter your email")
    phone = st.text_input("Enter your phone number")
    
    if st.button("REGISTER", key="register_btn_submit"):
        if password == confirm_password:
            cur.execute("SELECT * FROM users WHERE email=?", (email.lower(),))
            if cur.fetchone():
                st.error("User with this email already exists!")
            else:
                cur.execute("INSERT INTO users (name, password, email, phone) VALUES (?, ?, ?, ?)", 
                            (name, password, email.lower(), phone))
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
    cur = conn.cursor()
    name = st.text_input("User name")
    password = st.text_input("Password", type="password")
    
    if st.button("Login", key="login_btn_main"):
        cur.execute("SELECT * FROM users WHERE name=? AND password= ?", (name, password))
        user = cur.fetchone()
        if user:
            st.success(f"Welcome back, {user[1]}! Login successful!")
        else:
            st.error("Invalid user name or password!")
    
    if st.button("Forgot Password?", key="forgot_pwd_btn"):
        set_page("forgot_pwd")
    
    conn.close()

# ================ Forgot Password Page ===============
elif st.session_state.page == "forgot_pwd":
    st.markdown("<h1 style='text-align: center;'>Reset Password</h1>", unsafe_allow_html=True)
    
    conn = create_connection()
    cur = conn.cursor()
    username = st.text_input("Enter your username:")
    email = st.text_input("Enter your registered email:")
    
    if st.button("Send OTP", key="send_otp_btn"):
        cur.execute("SELECT * FROM users WHERE LOWER(name)=? AND LOWER(email)=?", (username.lower(), email.lower()))
        user = cur.fetchone()
        if user:
            otp = str(random.randint(100000, 999999))
            st.session_state.otp = otp
            st.session_state.reset_user = username
            
            sender_email = "your-email@gmail.com"
            sender_password = "your-app-password"
            
            try:
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender_email, sender_password)
                message = f"Subject: Password Reset OTP\n\nYour OTP is: {otp}"
                server.sendmail(sender_email, email, message)
                server.quit()
                
                st.success("OTP sent to your email.")
            except Exception as e:
                st.error(f"Failed to send OTP: {str(e)}")
        else:
            st.error("Email has not been registered.")
    
    if st.session_state.otp:
        otp_input = st.text_input("Enter OTP:", type="password")
        if st.button("Verify OTP", key="verify_otp_btn"):
            if otp_input == st.session_state.otp:
                new_password = st.text_input("New Password:", type="password")
                confirm_new_password = st.text_input("Confirm Password:", type="password")
                if st.button("Update Password", key="update_pwd_btn") and new_password == confirm_new_password:
                    cur.execute("UPDATE users SET password=? WHERE name=?", (new_password, st.session_state.reset_user))
                    conn.commit()
                    st.success("Password updated successfully!")
                    if st.button("Go to Login", key="go_to_login_btn"):
                        set_page("login")
            else:
                st.error("Invalid OTP!")
    conn.close()
