import streamlit as st
import requests
import json

# FastAPI endpoints
BASE_URL = "http://localhost:8000"  # Ensure this is the correct URL where FastAPI is running

# Streamlit UI Setup
st.set_page_config(page_title="ATS CV Search", page_icon="📄", layout="wide")

# Title and Description
st.title("ATS CV Search Interface")
st.markdown("""
Welcome to the **ATS CV Search** tool! 
You can search and analyze CVs using two methods:
- **Direct Elasticsearch Search**
- **Semantic Vector Search (using embeddings)**

Choose a method below to begin your search.
""")

# Sidebar for navigation
st.sidebar.title("ATS CV Search")
app_mode = st.sidebar.radio("Select a search method", ["Search CVs", "Vector Search", "Health Check"])

# Functions for making API calls
def search_cvs(query: str):
    """
    Perform a direct search using the Elasticsearch method.
    """
    response = requests.post(f"{BASE_URL}/search/", json={"query": query})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def vector_search_cvs(query: str):
    """
    Perform a semantic search using the vector search method.
    """
    response = requests.post(f"{BASE_URL}/vector-search/", json={"query": query})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def health_check():
    """
    Check the health of the FastAPI and Elasticsearch service.
    """
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Display mode content
if app_mode == "Search CVs":
    st.subheader("Search CVs using Elasticsearch")

    # Input text for search query
    query = st.text_input("Enter your search query:")

    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = search_cvs(query)
                if results and results.get("results"):
                    st.success("Search results found!")
                    st.json(results["results"])
                elif results:
                    st.warning("No results found.")
                else:
                    st.error("An error occurred while searching.")
        else:
            st.warning("Please enter a query to search.")

elif app_mode == "Vector Search":
    st.subheader("Search CVs using Semantic Vector Search")

    # Input text for vector search query
    query = st.text_input("Enter your search query for semantic search:")

    if st.button("Vector Search"):
        if query:
            with st.spinner("Searching..."):
                results = vector_search_cvs(query)
                if results and results.get("results"):
                    st.success("Search results found!")
                    st.json(results["results"])
                elif results:
                    st.warning("No results found.")
                else:
                    st.error("An error occurred while searching.")
        else:
            st.warning("Please enter a query to search.")

elif app_mode == "Health Check":
    st.subheader("Health Check - API & Elasticsearch Status")

    # Check system health
    with st.spinner("Checking health..."):
        status = health_check()
        if status:
            st.success("System is healthy!")
            st.write(f"API Status: {status['status']}")
            st.write(f"Elasticsearch: {status['elasticsearch']}")
        else:
            st.error("An error occurred while checking the health.")
