#!/usr/bin/env python
# coding: utf-8

# # L4: Tools for a Customer Outreach Campaign
# 
# In this lesson, you will learn more about Tools. You'll focus on three key elements of Tools:
# - Versatility
# - Fault Tolerance
# - Caching

# The libraries are already installed in the classroom. If you're running this notebook on your own machine, you can install the following:
# ```Python
# !pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
# ```

# In[ ]:


# Warning control
import warnings
warnings.filterwarnings('ignore')


# - Import libraries, APIs and LLM
# - [Serper](https://serper.dev)

# In[ ]:


from crewai import Agent, Task, Crew


# **Note**: 
# - The video uses `gpt-4-turbo`, but due to certain constraints, and in order to offer this course for free to everyone, the code you'll run here will use `gpt-3.5-turbo`.
# - You can use `gpt-4-turbo` when you run the notebook _locally_ (using `gpt-4-turbo` will not work on the platform)
# - Thank you for your understanding!

# In[ ]:


import os
# from utils import get_openai_api_key, pretty_print_result
# from utils import get_serper_api_key

from dotenv import load_dotenv

load_dotenv()  # Load the .env file


# ## Creating Agents

# In[ ]:


sales_rep_agent = Agent(
    role="Sales Representative",
    goal="Identify high-value leads that match "
         "our ideal customer profile",
    backstory=(
        "As a part of the dynamic sales team at AF Group, "
        "your mission is to scour "
        "the corporate landscape for potential leads. "
        "Armed with cutting-edge tools "
        "and a strategic mindset, you analyze data, "
        "trends, and interactions to "
        "unearth opportunities that others might overlook. "
        "Your work is crucial in paving the way "
        "for meaningful engagements and driving the company's growth."
    ),
    allow_delegation=False,
    verbose=True
)


# In[ ]:


insurance_expert = Agent(
    role="AF Group Workers Compensation Insurance Expert",
    goal="Ground sales team with insurance knowledge and AF Group knowledge",
    backstory=(
        "You're a workers compensation insurance expert at AF Group, "
        "your mission is to guide the sales team with your expertise "
        "and help them to reframe their sales pitches with accurate knowledge of AF Group insurance. "
        "Armed with knowledge "
        "and an insurance mindset, you analyze sales pitches, "
        "wording, and language to "
        "unearth errors that the sales team might overlook "
        "and correct promises that AF Group can't deliver "
        "and ensure that the sales team is selling what AF Group can offer."
    ),
    allow_delegation=False,
    verbose=True
)


# In[ ]:


lead_sales_rep_agent = Agent(
    role="Lead Sales Representative",
    goal="Nurture leads with personalized, compelling communications",
    backstory=(
        "Within the vibrant ecosystem of AF Group's sales department, "
        "you stand out as the bridge between potential clients "
        "and the insurance they need."
        "By creating engaging, personalized messages, "
        "you not only inform leads about our offerings "
        "but also make them feel seen and heard."
        "Your role is pivotal in converting interest "
        "into action, guiding leads through the journey "
        "from curiosity to commitment."
    ),
    allow_delegation=False,
    verbose=True
)


# ## Creating Tools
# 
# ### crewAI Tools

# In[ ]:


from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool


# In[ ]:


# directory_read_tool = DirectoryReadTool(directory='./instructions')
# file_read_tool = FileReadTool()
# search_tool = SerperDevTool()


# ### Custom Tool
# - Create a custom tool using crewAi's [BaseTool](https://docs.crewai.com/core-concepts/Tools/#subclassing-basetool) class

# In[ ]:


from crewai_tools import BaseTool


# - Every Tool needs to have a `name` and a `description`.
# - For simplicity and classroom purposes, `SentimentAnalysisTool` will return `positive` for every text.
# - When running locally, you can customize the code with your logic in the `_run` function.

# In[ ]:


class SentimentAnalysisTool(BaseTool):
    name: str ="Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of text "
         "to ensure positive and engaging communication.")
    
    def _run(self, text: str) -> str:
        # Your custom code tool goes here
        return "positive"


# In[ ]:


sentiment_analysis_tool = SentimentAnalysisTool()


# ## Creating Tasks
# 
# - The Lead Profiling Task is using crewAI Tools.

# In[ ]:


lead_profiling_task = Task(
    description=(
        "Conduct an in-depth analysis of {lead_name}, "
        "a company in the {industry} sector "
        "that recently showed interest in our insurance. "
        "Utilize all available data sources "
        "to compile a detailed profile, "
        "focusing on key decision-makers, recent business "
        "developments, and potential needs "
        "that align with our offerings. "
        "This task is crucial for tailoring "
        "our engagement strategy effectively.\n"
        "Don't make assumptions and "
        "only use information you absolutely sure about."
    ),
    expected_output=(
        "A comprehensive report on {lead_name}, "
        "including company background, "
        "key personnel, recent milestones, and identified needs. "
        "Highlight potential areas where "
        "our insurance can provide value, "
        "and suggest personalized engagement strategies."
    ),
    # tools=[directory_read_tool, file_read_tool, search_tool],
    agent=sales_rep_agent,
)


# - The Personalized Outreach Task is using your custom Tool `SentimentAnalysisTool`, as well as crewAI's `SerperDevTool` (search_tool).

# In[ ]:


personalized_outreach_task = Task(
    description=(
        "Using the insights gathered from "
        "the lead profiling report on {lead_name}, "
        "craft a personalized outreach campaign "
        "aimed at {key_decision_maker}, "
        "the {position} of {lead_name}. "
        "The campaign should address their recent {milestone} "
        "and how our insurance can support their goals. "
        "Your communication must resonate "
        "with {lead_name}'s company culture and values, "
        "demonstrating a deep understanding of "
        "their business and needs.\n"
        "Don't make assumptions and only "
        "use information you absolutely sure about."
    ),
    expected_output=(
        "A series of personalized email drafts "
        "tailored to {lead_name}, "
        "specifically targeting {key_decision_maker}."
        "Each draft should include "
        "a compelling narrative that connects our insurance "
        "with their recent achievements and future goals. "
        "Ensure the tone is engaging, professional, "
        "and aligned with {lead_name}'s corporate identity."
    ),
    tools=[sentiment_analysis_tool], # search_tool
    agent=lead_sales_rep_agent,
)

# In[ ]:


insurance_editing = Task(
    description=(
        "Review the latest sales pitch from our sales team, "
        "and ensure that the sales pitch is grounded with AF Group offerings "
        "and make sure that the pitch is offering insurance. "
        "Utilize all available data sources "
        "to compile a detailed revision, "
        "focusing on accurate insurance workers comp information, recent AF Group business "
        "developments, and offerings "
        "that align with AF Group. "
        "This task is crucial for tailoring "
        "our engagement to ensure it aligns with AF Group.\n"
        "Don't make assumptions and "
        "only use information you absolutely sure about."
    ),
    expected_output=(
        "Revisions of all personalized email drafts, "
        "grounded on AF Group offerings. "
    ),
    # tools=[directory_read_tool, file_read_tool, search_tool],
    agent=insurance_expert,
)


# ## Creating the Crew

# In[ ]:


crew = Crew(
    agents=[sales_rep_agent, 
            lead_sales_rep_agent],
    
    tasks=[lead_profiling_task, 
           personalized_outreach_task,insurance_editing,personalized_outreach_task],
	
    verbose=True,
	memory=True
)


# ## Running the Crew
# 
# **Note**: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.

# In[ ]:


inputs = {
    "lead_name": "Lake Michigan Construction Company",
    "industry": "Construction",
    "key_decision_maker": "Kari Noyola",
    "position": "CEO",
    "milestone": "Record new home building sales"
}

result = crew.kickoff(inputs=inputs)


# - Display the final result as Markdown.

# In[ ]:


# from IPython.display import Markdown
# Markdown(result)


# In[ ]:

print("----------------------------------------------------------")
print("-----------------------RESULT-----------------------------")
print("----------------------------------------------------------")
print(result)


# In[ ]:




