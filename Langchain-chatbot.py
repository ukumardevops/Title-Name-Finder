# https://colab.research.google.com/drive/1qel8IBkCO3iW-qOShz3oJFxCgjo77gSG?usp=sharing#scrollTo=Eb9JV3tGPdm5

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
import warnings
from dotenv import load_dotenv
import os
from pprint import pprint
from IPython.display import Markdown
import textwrap

load_dotenv()

warnings.filterwarnings("ignore")

# Information Retrival
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.0)

context = ""

prompt_template = """You are an expert in finding the English Title Name for given input title of movie, TV Show, Bundle Title names in any language as precise as \
                     possible using the imdb dataset. If the given title is TV show then provide Season number and episode number as well. \
                     Also, provide the release year as part of output in case of movie title. \
                     And if the movie, TV Show, Bundle Title names are not found then say "English Title Not Found". \
                     For Example: \
                     if the input title: "sp man 2" then English Title Name: "Spider-Man 2(2004)". \n\n
                     if the input title: "Real Ghostbusters - Hole in the Wall Gang" then English Title Name: "The Real Ghostbusters: S02 E65 · The Hole in the Wall Gang"
                     Context: \n {context}?\n
                     input title: \n {question} \n
                     English Title Name:
                  """

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

question = "sp mn nwh"

stuff_answer = stuff_chain(
    {"input_documents": "", "question": question}, return_only_outputs=True
)

print("input title: " + question)
pprint(stuff_answer)

"""
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# Text Generation
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-pro")

prompt = [
    "What is Mixture of Experts?",
]

response = model.generate_content(prompt)
to_markdown(response.text)
"""