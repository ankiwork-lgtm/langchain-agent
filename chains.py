from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral X influencer grading a X post. Generate critique and recommendation for the user"
            "Always provide detailed recommendations, including requests for length, virality, style, etc. "
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a X techie influencer assistant tasked with writing excellent X posts"
            "Generate the best X post possible forthe user's request"
            "If the user provides critique, respond with a revised version of your previous attemps."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm