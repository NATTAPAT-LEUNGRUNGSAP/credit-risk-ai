import streamlit as st

st.set_page_config(
    page_title="Credit risk AI",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/home:"),
    st.Page("app_pages/applicant.py", title="Applicant explorer", icon=":material/person:"),
    st.Page("app_pages/portfolio.py", title="Portfolio", icon=":material/bar_chart:"),
    st.Page("app_pages/simulator.py", title="Business simulator", icon=":material/tune:"),
    st.Page("app_pages/model.py", title="Model insights", icon=":material/analytics:"),
]

page = st.navigation(pages, position="top")
st.title(page.title, icon=page.icon)
page.run()

st.caption(
    "Educational credit-risk decision-support demo. Not an automated approval system, "
    "credit policy, or substitute for human and regulatory review."
)

