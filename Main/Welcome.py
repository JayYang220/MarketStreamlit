import streamlit as st
from API import StockManager
import os

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data():
    manager = StockManager(abs_path=os.path.dirname(os.path.abspath(__file__)))
    return manager

st.set_page_config(
    page_title="Welcome!",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://blog.jiatool.com/posts/streamlit_2023/',
        'About': ""
    }
)

manager = load_data()
st.title("Welcome!")
st.subheader("Please select a function from the sidebar.")


# menu()  # Render the dynamic menu!
