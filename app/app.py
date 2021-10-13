import streamlit as st
import pandas as pd
# import database_func as db
import app_func as af


def main():

    st.sidebar.title('BIRD SONGS WIKI')
    af.classification()


if __name__ == '__main__':
    main()
