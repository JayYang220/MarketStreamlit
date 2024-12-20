import streamlit as st
from Welcome import manager

stock_list = manager.get_stock_list()
output_list = []

if stock_list:
    # Verify the user's role
    st.subheader("Analysis:")
    with st.form(key='Analysis'):
        form_gender = st.selectbox('Please select a item:', manager.get_stock_list())
        submit_button = st.form_submit_button(label='Show Data')

        if submit_button:
            manager.show_data(form_gender)

else:
    st.subheader("There is no data in your repositories. Please download first.")
