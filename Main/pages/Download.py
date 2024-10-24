import os
import time

import streamlit as st
from menu import menu_with_redirect
from API import StockManager
from Welcome import manager

def action_download(function, *args, output_list):
    output_list.append(st.empty())
    output_list[-1].write("###### Downloading...")

    # update 1 stock
    if len(args) != 0:
        for arg in args:
            function(arg)
        for msg in manager.msg:
            output_list[-1].write(f"###### {msg}")

    # update many stock
    else:
        function()
        for msg in manager.msg:
            output_list.append(st.empty())
            output_list[-1].write(f"###### {msg}")

    return manager.is_action_successful

def clear_output_list(output_list):
    if output_list != []:
        for i in output_list:
            i.empty()


stock_list = manager.get_stock_list()
output_list = []

if stock_list:
    # Verify the user's role
    st.subheader("Update History Data:")
    with st.form(key='history'):
        form_gender = st.selectbox('Please select a item to update the data:', manager.get_stock_list())

        submit_button = st.form_submit_button(label='Submit')
        submit_button2 = st.form_submit_button(label='Update All')

        if submit_button:
            clear_output_list(output_list)
            action_download(manager.update_history, form_gender, output_list=output_list)

        if submit_button2:
            clear_output_list(output_list)
            action_download(manager.update_all, output_list=output_list)
else:
    st.subheader("There is no data in your repositories. Please add first.")


st.subheader("Add stock:")
with st.form(key='add'):
    form_name = st.text_input(label='Enter the stock name', placeholder='Enter the stock name')

    submit_button3 = st.form_submit_button(label='Submit')
    st.markdown("You can retrieve stock names from [Yahoo Finance](https://finance.yahoo.com/).")

    if submit_button3:
        clear_output_list(output_list)
        is_action_successful = action_download(manager.create_stock_class, form_name, output_list=output_list)
        if is_action_successful:
            time.sleep(1)
            st.rerun()

if stock_list:
    st.subheader("Remove stock:")
    with st.form(key='Remove'):
        form_gender = st.selectbox('Please select a item to update the data:', manager.get_stock_list())

        submit_button4 = st.form_submit_button(label='Submit')

        if submit_button4:
            clear_output_list(output_list)
            is_action_successful = action_download(manager.remove_stock, form_gender, output_list=output_list)
            if is_action_successful:
                time.sleep(1)
                st.rerun()
