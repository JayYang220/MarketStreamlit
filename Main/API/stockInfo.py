# Guide https://pypi.org/project/yfinance/
# Guide https://algotrading101.com/learn/yahoo-finance-api-guide/
import yfinance as yf
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from common import debug_msg

is_debug = False

class StockManager:
    __slots__ = ["history_data_folder_path", "stock_name_list", "stock_class_list", "msg", "is_action_successful"]

    def __init__(self, abs_path):
        self.history_data_folder_path = os.path.join(abs_path, "data")

        # 讀取庫存的CSV名稱 集中成list管理
        self.stock_name_list = self.__init_load_stock_list()

        # 建立stock class 集中成list管理
        self.stock_class_list = self.__init_create_stock_class()

        self.msg = []
        self.is_action_successful = False

    def __init_load_stock_list(self):
        """讀取庫存的CSV名稱 集中成list管理"""
        if os.path.exists(self.history_data_folder_path) is False:
            # 從未下載過任何資料時
            os.mkdir(self.history_data_folder_path)
            return []
        else:
            # 讀取並返回
            file_list = os.listdir(self.history_data_folder_path)
            file_list_without_extension = [os.path.splitext(file)[0] if file.endswith('.csv') else file for file in file_list]
            return file_list_without_extension

    def __init_create_stock_class(self):
        """建立stock class 集中成list管理"""
        stock_class_list = []
        for stock_name in self.stock_name_list:
            stock_class_list.append(Stock(self.history_data_folder_path, stock_name))
        return stock_class_list

    def create_stock_class(self, stock_name: str):
        """try create"""
        self.msg.clear()
        if stock_name in self.stock_name_list:
            self.is_action_successful = False
            debug_msg(is_debug, "This stock already exists in the list.")
            self.msg.append("This stock already exists in the list.")
            return

        # try downloading ticker
        try:
            stock = Stock(self.history_data_folder_path, stock_name)

            if stock.download_ticker():
                stock.download_history_data()

                debug_msg(is_debug, f"{stock_name} Addition completed")
                self.msg.append(f"{stock_name} Addition completed")

                self.stock_class_list.append(stock)
                self.stock_name_list.append(stock_name)
                self.is_action_successful = True
            else:
                self.msg.append(f"{stock_name}: Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/.")
                self.is_action_successful = False

        except Exception as e:
            self.is_action_successful = False
            debug_msg(is_debug, e)
            self.msg.append(f"{stock_name}: {e}")
            return

    def remove_stock(self, stock_name: str):
        try:
            self.msg.clear()
            stock_index = self.stock_name_list.index(stock_name)
            path = self.stock_class_list[stock_index].history_data_file_path
            os.remove(path)

            self.stock_name_list.remove(stock_name)
            self.stock_class_list.pop(stock_index)
            self.msg.append(f"{stock_name}: Remove completed.")

            self.is_action_successful = True
        except Exception as e:
            self.msg.append(f"{stock_name}: {e}")
            self.is_action_successful = False

    def update_history(self, stock_name):
        self.msg.clear()
        stock_index = self.stock_name_list.index(stock_name)
        self.msg.append(self.stock_class_list[stock_index].download_history_data())
        self.is_action_successful = True

    def update_all(self):
        """更新所有股票資訊"""
        self.msg.clear()
        if self.stock_class_list:
            for stock in self.stock_class_list:
                self.msg.append(stock.download_history_data())
            self.is_action_successful = True
        else:
            debug_msg(is_debug, "There is no data in the system.")
            self.msg.append("There is no data in the system.")
            self.is_action_successful = False

    def get_stock_list(self):
        """顯示庫存的CSV名稱"""
        if self.stock_name_list:
            return self.stock_name_list
        else:
            debug_msg(is_debug, "No Data.")
            self.msg.append("No Data.")
            return False

    def get_company_info(self, stock_name):
        self.msg.clear()
        stock_index = self.stock_name_list.index(stock_name)
        try:
            if self.stock_class_list[stock_index].get_company_info():
                info = self.stock_class_list[stock_index].company_info

                self.msg.append(info)
                debug_msg(is_debug, info)
                self.is_action_successful = True
                return info
            else:
                self.is_action_successful = False

        except Exception as e:
            self.msg.append(f"{stock_name}: {e}")
            self.is_action_successful = False

    def show_stock_list(self):
        """顯示庫存的CSV名稱"""
        if self.stock_name_list:
            for index, stock in enumerate(self.stock_name_list):
                print(f"{index:>3d}. {stock}")
        else:
            print("No Data.")
    
    def get_analysis(self, stock_name):
        stock_index = self.stock_name_list.index(stock_name)
        return self.stock_class_list[stock_index].test()
    
    def show_data(self, stock_name):
        stock_index = self.stock_name_list.index(stock_name)
        return self.stock_class_list[stock_index].show_data()


class Stock:
    __slots__ = ["stock_name", "history_data_file_path", "ticker", "company_info", "history_data", "data"]

    def __init__(self, history_data_folder_path, stock_name: str):
        self.stock_name = stock_name
        self.history_data_file_path = os.path.join(history_data_folder_path, self.stock_name + ".csv")
        self.data = None

        # 初始化為None，待使用者輸入需求時再抓取
        self.ticker = None
        self.company_info = None
        self.history_data = None

    def download_ticker(self):
        """下載 ticker"""

        self.ticker = yf.Ticker(self.stock_name)

        # 股票名稱錯誤時，仍會返回一個dict，利用下列特徵確認股票名稱是否正確
        if 'previousClose' not in self.ticker.info:
            # raise AssertionError("Stock name error.")
            self.ticker = False
            return False
        else:
            return self.ticker

    def get_company_info(self):
        self.download_ticker()
        if self.ticker:
            self.company_info = self.ticker.info
        return self.ticker

    def show_company_info(self):
        """顯示 CompanyInfo"""
        if self.ticker is None:
            self.ticker = self.download_ticker()
            if self.ticker is False:
                return

        if self.company_info is None:
            self.company_info = self.ticker.info

        for key in self.company_info.keys():
            debug_msg(is_debug, f"{key:30s} {self.company_info[key]}")

    def show_history_data(self):
        """顯示 HistoryData"""
        if self.history_data is None:
            self.load_history()
        debug_msg(is_debug, self.history_data)

    def load_history(self):
        """讀取 HistoryData"""
        try:
            self.history_data = pd.read_csv(self.history_data_file_path)
        except Exception as e:
            debug_msg(is_debug, e)
            return e

    def download_history_data(self, period: str = 'max', interval: str = '1d'):
        """Download HistoryData"""

        self.ticker = self.download_ticker()

        try:
            if self.ticker:
                self.history_data = self.ticker.history(period=period, interval=interval)
                self.history_data['Date'] = pd.to_datetime(self.history_data.index).strftime('%Y-%m-%d')
                self.history_data.to_csv(self.history_data_file_path, index=False)
                debug_msg(is_debug, f"{self.stock_name}: Update completed.")
                return f"{self.stock_name}: Update completed."
            else:
                debug_msg(is_debug, "Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/")
                return f"{self.stock_name}: Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/."
        except Exception as e:
            return f"{self.stock_name}: {e}"
    
    def show_data(self):
        self.data = pd.read_csv(self.history_data_file_path)
        show_type_list = ["Open", "High", "Low", "Close", "Volume"]

        for column in show_type_list:        
            # 創建Plotly圖表
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data["Date"], y=self.data[column], mode='lines', name=column))
        
        # 設置圖表佈局
            fig.update_layout(
                title=f"{self.stock_name} {column}",
                xaxis_title="日期",
                yaxis_title=f"{column} (USD)",
                xaxis_rangeslider_visible=True
            )
            # 使用st.plotly_chart顯示圖表
            st.plotly_chart(fig)

