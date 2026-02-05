import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

print("Initializing components....")

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
    )


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a string format"""
    return "\n\n".join([doc.page_content for doc in docs])

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieve documents and format them.
    """
    #step1: retrieve documents
    docs = retriever.invoke(query)
    #step2: format documents
    context = format_docs(docs)
    #step3: Format the prompt with the context
    messages = prompt_template.format_messages(context=context, question=query)
    #step4: Invoke the LLM with formatted messages
    response = llm.invoke(messages)
    #step5: return the response
    return response.content

def retrieval_chain_with_lcel():
    """
    Create a retrieval chain with LCEL(Langchain Expression Language)
    Returns a chain that can be invoked with {{"question": "..."}}   
    No input is required as input is through invoke
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "what is pinecone in machine learning?"


    # Use implementation without LCEL
    print("\n" + "=" *50)
    print("Implementation without LCEL")
    print("\n" + "=" *50)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # Use implementation with LCEL
    print("\n" + "=" *50)
    print("Implementation with LCEL")
    print("\n" + "=" *50)
    print("More concise and declaration")
    print("Built in streaming: chain.stream()")
    print("Built-in async: chain.ainvoke")
    print("Easy to compose with other chains")
    print("Better for production use")
    print("="*50)
    chain_with_lcel = retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print(result_with_lcel)
