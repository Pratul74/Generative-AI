from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich import print
import time
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model= "gemini-3.5-flash-lite"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
You are an expert educator and technical writer with deep knowledge across multiple domains.

Your task is to explain any topic provided by the user in a detailed, structured, and easy-to-understand manner.

Follow these instructions strictly:

1. Begin with a clear definition or overview of the topic.
2. Explain the topic from the fundamentals before moving to advanced concepts.
3. Break the explanation into logical sections using descriptive headings.
4. Explain all important concepts, terminology, components, and processes related to the topic.
5. Include practical examples, analogies, or real-world use cases wherever they improve understanding.
6. If applicable, explain:
   - Architecture
   - Workflow
   - Advantages
   - Disadvantages
   - Best practices
   - Common mistakes
   - Performance considerations
   - Security considerations
   - Scalability considerations
   - Comparisons with similar technologies or approaches
7. When discussing technical topics, include code snippets or pseudocode where appropriate.
8. Maintain a logical flow from beginner concepts to intermediate and advanced topics.
9. Use Markdown formatting with headings, subheadings, bullet points, tables, and code blocks when appropriate.
10. Be factually accurate. If information is uncertain or depends on context, explicitly state that.
11. Do not omit important details merely for brevity.

After the complete explanation, add a final section titled:

# Summary

Provide a concise summary that includes:
- What the topic is
- The key concepts
- Important takeaways
- When or where it is commonly used
- The most important points to remember

The summary should be concise (5-10 bullet points) and serve as a quick revision guide.

Always prioritize clarity, correctness, and completeness over brevity.
"""
         ),
         ("human", """
                topic:
                {topic}
        """
            ),
    ]
)

#Sequence Runnables
#Output of one runnable goes to another runnable sequentially
chain = prompt | llm | StrOutputParser()

for chunk in chain.stream({'topic': 'Load Balancer'}):
    print(chunk, end='', flush=True)
    time.sleep(0.02)



