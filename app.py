import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
key_value = os.getenv("OPENAI_API_KEY")
if key_value is None or key_value == "":
    print("OPENAI_API_KEY is not set in DEV")
    exit(1)
else:
    print("OPENAI_API_KEY is set in DEV")


#By using st.set_page_config(), you can customize the appearance of your Streamlit application's web page
st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

#Initialize the OpenAIEmbeddings object
embeddings = OpenAIEmbeddings()

#The below snippet helps us to import CSV file data for our tasks
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})

#Assigning the data inside the csv to our variable here
data = loader.load()

#Display the data
print(data)

db = FAISS.from_documents(data, embeddings)

#Function to receive input from user and store it in a variable
def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text


user_input=get_text()
submit = st.button('Find similar Things')  

if submit:
    
    #If the button is clicked, the below snippet will fetch us the similar text
    docs = db.similarity_search(user_input)
    #docs = db.similarity_search_with_score(user_input)
    st.subheader("Top Matches:")
    st.text(docs) 
    for doc in docs:
        st.text(doc.page_content) 
    # st.text(docs[0])  
    # st.text(docs.page_content)

