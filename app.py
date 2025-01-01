from langgraph.graph import StateGraph, START, END 
from pydantic import BaseModel
from groq import Groq
from typing import List, Dict, Literal, Annotated, Optional
from typing_extensions import TypedDict
from IPython.display import Image, display
from operator import add
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from transformers import pipeline
from langchain.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain.chains import LLMChain   
import streamlit as st
# import tensorflow as tf
import time
import pandas as pd
import json
import os
from rough import remove_code
from schemas import InputState, OutputState, OverallState, Agent
from classes import Task
#, SubTask, SubSubTask
from tools import code_runner
#, create_text_embeddings, semantic_search, translate, web_search, scrapper, youtube_search, get_video_content
from dotenv import load_dotenv
load_dotenv()

st.title("PrismAI - A collaborative AI agentic assistant")
chat_input = st.chat_input("Ask me something. Eg. Give a summary of this article in hindi", key="chat_input")
uploaded_files = st.file_uploader("Upload one or more files", accept_multiple_files=True)






model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
    # other params...
)


groq = Groq()










prompt_enhancer_plan="""
Act as a problem understanding specialist. Understand the use's problem and give a detailed and descriptive prompt which will be used to create some ai agents for each sub-task. The prompt should include the overall task user is asking to do, then it should include all possible subtasks in order. All subtasks should be clear, specific and very detailed. The sub-task description should contain the sub-problem to solve.
Don't specify the tools or llms to use, just specify the sub-tasks. don't divide the subtasks further into sub-sub-tasks. Don't create redundant sub-tasks. Example :- 'load the csv file into dataframe' is a redundant sub task because it have to be done everytime whenever the agent wants to do any operation on the csv file. Instead create a subtask which is solving a specific problem or may providing some output for user's analysis.
If the user's query is not clear or the task needs some user inputs during subtasks, don't randomly assume the preferences, in this case include 'asking for more information and clarity about exactly what the user want' in the sub-tasks.

available llms :- 

1. coder 
2. vision (image analyser model)
3. gemini (multipurpose language model)
4. summarization

available tools :- 
1. web_search 
2. scrapper 
3. youtube_search 
4. get_video_content 
5. create_text_embeddings 
6. semantic_search 
7. translate 
8. code_runner

Above tools and llms will be collaboratively used to solve a subtask in next step. but don't specify them right now.

Some tips and examples:-
if task needs some recent information then first subtask should be to create web search queries and search on web.
If user wants to do document Q&A, then first subtask will be to create embeddings from document, second will be analysing the user's query to find out keywords to search, third will be to search from the documents and gather relevant information and give a final summary
Generation of code and running of same code should fall into one sub task. Though there might be different subtasks for different codes.
Return a json object with the following schema:-
{{  

    "task": str (overall task description),
    "inputs": Dict[str,str] (json containing all inputs given by the user. example:- csv_file_location : 'path/to/file.csv') (Return empty dict if no inputs given),
    "sub_tasks": List[Dict] (list of jsons, each json having id (int) (1/2/3..) ,subtask_description (str) and previous_knowledge (list of ids of previous subtasks whose outputs are necessary for this subtask. Empty list if no previous knowledge required)),

}}
Give output with no preambles or postambles. Keep all strings in double quotes\n

"""

prompt_enhancer_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_enhancer_plan,
        ),
        (
            "human",
            (
                "{question}"
            ),
        ),
    ]
)
print("prompt enhancer prompt\n",prompt_enhancer_prompt,"\n")
enhancer_chain= prompt_enhancer_prompt | model | StrOutputParser()
def prompt_enhancer_node(state: str) -> str:
    prompt_enhancer = enhancer_chain.invoke({"question": state})
    if prompt_enhancer[0]=="`":
        prompt_enhancer=prompt_enhancer[7:-4]
        prompt_enhancer=json.loads(prompt_enhancer)
    else:
        prompt_enhancer=json.loads(prompt_enhancer)
    # prompt_enhancer["inputs"]=json.loads(prompt_enhancer["inputs"])
    # prompt_enhancer["sub_tasks"]=json.loads(prompt_enhancer["sub_tasks"])
    print("ENHANCED PROMPT :-",prompt_enhancer,"\n")
    st.write("-" * 20)
    with st.chat_message("ai"):
        st.write(f"Step: prompt_enhancer")
        st.write(f"Prompt: ")
        st.json(prompt_enhancer)
    return prompt_enhancer
    







def rational_plan_node(state: str) -> Agent:
    chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
As an intelligent planning agent, your objective is to assign the subtasks to each agent. for each sub-task give the following:-
1. id : (int) 1/2/3...
2. sub_task : (str) description of sub task
3. description :{{'type' : tool / llm, 'name' : tool name / llm name}}   (json containing type and name)
4. prompt : (str) very detailed prompt specific to agent's sub-task. Give none if the agent type is tool.
5. input: {{A : i1, B : i2}} (json containing definition of all inputs required by the agent to perform it's task. Let i1 be any output of any previous agent p (p is agent id starting from 1), hence write output_p in place of i1 to get all outputs of agent p. STRICTLY MAKE SURE THAT output_0 does not exists! (please specify output_p.code or output_p.language or output_p.anything where code ,language and anything are more specific outputs in output_p). Let i2 be the input which can be constructed now using the known information, hence construct the required i2. write actual input names instaed of A, B. I used these names for explanation. The names should be big and unique to avoid confusions.)
6. extra_input : List[str] (List of questions that will be asked to the user. Question should include all additional information required from the user for more knowledge about the task. Keep it empty if no extra information is required.)
7. output: [o1, o2, o3] List[str] (list of definition of outputs the agent will produce. Example a code_runner agent will output the logs, path of any saved image file / csv file. write actual output names instaed of o1, o2, o3. I used these names for explanation. Keep the names unique to avoid confusions.)

Don't keep redundant inputs and extra inputs, which is already known and can be directly included in the prompt.
The inputs of first agent are known. hence don't write output_0, instead construct the input and write the actual input.

available llms :- 
1. coder :  code generation. input : prompt, output : code and language 
2. vision : image analyser. input : image_location, prompt; output : description
3. gemini : generalized multipurpose llm. input : question, output : generated output in structured json format only  (use this to find search queries {{"queries" : ["q1","q2"]}}, creating structured outputs from user's unstructured inputs)
4. summarization : Specialized in summarizing large text. input : prompt containing logs or textual information (it should not contain csv file or image files), output : summary text

available tools (these are actually functions with well defined inputs and outputs.) :- 
1. web_search : website searcher. input : search_query, output : web_links (list)
2. scrapper : web scrapper. input : web_links (list), output : scrapped_data
3. youtube_search : youtube video search. input : search_query,output : video_links (list)
4. get_video_content : given a video, get the transcript. input : video_link (list) or video_location (if saved in device). output : transcript
5. create_text_embeddings : divide any text into chunks and create embeddings. input : document_location, output : created
6. semantic_search : search similar text using embeddinngs. input : search_query, output : top_k_results
7. translate : translate text to other language. inputs : text, destnation_language_code; output : translated_text. (destination_language_code : 'en' for english, 'de' for german, 'hi' for hindi, 'fr' for french)
8. code_runner : run the code and give the output. inputs : code_language, code generated by coder agent (llm); output : output of code


Return a list of jsons, each json for each sub-task (agent). The python list should be arranged in order of agents to be used. 
Strictly keep ONLY THE AVAILABLE slms/llms and tools mentioned above in the output! 


Some Tips:-
if the task is to generate code, then the workflow should be STRICTLY: coder llm (to generate the code) -> code_runner tool (to run the code)-> summarization llm (to summarize the output logs). The code must produce some analysis or output logs which can be summarized by summarization llm. 
The runtime variables of one code cannot be shared to the other, hence prompt for coder agent should be STRICTLY COMPLETE IN ITSELF! such that the generatd code's output provides visible results.  i.e. don't divide it futher into multiple coder agents for different parts of the code.
if the task is document Q&A, then first use create_text_embeddings and then use semantic_search.
if task needs some recent information then first keep web_search and scrapper
Before each search tool agent, keep gemini llm to find the small and specific search query.
The last agent should be summarization llm to summarize and give conclusion and analysis about the overall task. For this agent, the prompt should be written such that summary should include all minute details. This agent can also give suggestions for further analysis or tasks based on the results if required. It will only have one output ["summary"].


Give output with no preambles or postambles. Keep all strings in double quotes.
"""
            },
            {
                "role": "user",
                "content": state,
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,    
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format
    )
    print("AGENTS NEEDED:-",chat_completion.choices[0].message.content,"\n")
    ans=json.loads(chat_completion.choices[0].message.content)
    # for i in ans:
        # i["input"]=json.loads(i["input"])
        # i["output"]=json.loads(i["output"])
        # i["extra_input"]=json.loads(i["extra_input"])
        # i["description"]=json.loads(i["description"])
        # i["id"]=int(i["id"])
    st.write("-" * 20)
    with st.chat_message("ai"):
        st.write(f"Step: rational_plan")
        st.write(f"Rational plan: ")
        st.json(ans)
    return ans















code="""You are a coder agent. The code should be generated using the given prompt by human. Import all the libraries first. The code should be error free with proper exception handling. The code should save all new files formed or changed files in directory TEMP/NEW/. Eg :- box plots created during execution or changed dataframes in csv, with unique and big and readable names to avoid confusions.
Add print statement after all snippets where the output logs might be useful for the user or next llm agents to analyse the results. The print statements must include what is being printed.
generate python code by default if not specified by the user. If the user wants code in other languages, then the user will specify the language in the prompt. The code should be generated in the specified language. No preambles or postambles are required. Generate only the code with proper comments.

Output format:- JSON object with the following schema:-
{{
    "code": str (generated code),
    "language": str (language of the code)
}}
"""

coder_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            code,
        ),
        (
            "human",
            (
                "{prompt}"
            ),
        ),
    ]
)

coder_chain= coder_prompt| model | StrOutputParser()

gemini_prompt=ChatPromptTemplate.from_messages(
    [
        
        (
            "human",
            (
                "{prompt}"
            ),
        ),
    ]
)
gemini_chain= gemini_prompt| model | StrOutputParser()
def process_subtask(agent_list,sub_task,notebook):
    global code
    sub_note={}
    idx=sub_task["id"]
    desc=sub_task["subtask_description"]
    sub_note["subtask_description"]=desc
    for agent in agent_list:
            print(f"This is Agent {agent['id']} , \nSUB-SUB-TASK :- {agent['sub_task']}")
            type=agent["description"]["type"]
            name=agent["description"]["name"]
            with st.chat_message("ai"):
                st.write(f"Step: PROCESSING AGENT {agent['id']}:-\n{agent['sub_task']}")
            print("Agent type:-",type)
            print("Agent name:-",name)
            
                    
            if type=="llm":
                if agent['extra_input']:
                    st.write(f"EXTRA INPUTS REQUIRED:-")
                    print("EXTRA INPUTS REQUIRED:-")
                    for q in agent["extra_input"]:
                        e_inp=input(q)
                        agent["input"][q]=e_inp
                    
                for i in agent['input']:    
                    if agent['input'][i][:7]=="output_":
                        try:
                            agent['input'][i]=sub_note[agent['input'][i][:8]][agent['input'][i][9:]]
                        except:

                            agent['input'][i]=sub_note[agent['input'][i][:8]]

                print("INPUTS GOING IN PROMPT:-",agent['input'])
                with st.chat_message("user"):
                    st.write("Inputs going into prompt:-")
                    st.json(agent["input"])

                if name=="coder":
                    cod=agent["prompt"]
                    print("CODE PROMPT:-",code)
                    st.write(f"CODE PROMPT:-\n{code}\n{cod} and some additional information:- ")
                    st.json(agent['input'])
                    code_generated=coder_chain.invoke({"prompt": cod+f"\nAdditional information:-\n {agent['input']}"})
                    print("CODE GENERATED:-",code_generated)
                    if code_generated[0]=="`":
                        code_generated=code_generated[7:-4]
                        code_generated=json.loads(code_generated)
                    else:
                        code_generated=json.loads(code_generated)
                    st.code(code_generated['code'],language=code_generated['language'])
                    
                    sub_note[f"output_{agent['id']}"]=code_generated
                elif name=="gemini":
                    cod=agent["prompt"]
                    print("GEMINI PROMPT:-",cod)
                    st.write(f"GEMINI PROMPT:-\n{cod} and some additional information:-\n")
                    st.json(agent['input'])
                    out=agent["output"]
                    req_out=f"Required outputs:-\n {out}"
                    gemini_generated=gemini_chain.invoke({"prompt": cod+f"\nAdditional information:-\n {agent['input']}\n"+req_out+"\n give output in json format only. no preambles or postambles"})
                    print("GEMINI GENERATED:-",gemini_generated)
                    st.write(f"GEMINI GENERATED:-")
                    if gemini_generated[0]=="`":
                        gemini_generated=gemini_generated[7:-4]
                        gemini_generated=json.loads(gemini_generated)
                    else:
                        gemini_generated=json.loads(gemini_generated)
                    st.json(gemini_generated)
                    sub_note[f"output_{agent['id']}"]=gemini_generated
                    
                elif name=="summarization":
                    cod=agent["prompt"]
                    print("SUMMARIZATION PROMPT:-",cod)
                    st.write(f"SUMMARIZATION PROMPT:-\n{cod} and some additional information:-\n")
                    st.json(agent['input'])
                    out=agent["output"]
                    req_out=f"Required outputs:-\n {out}"
                    gemini_generated=gemini_chain.invoke({"prompt": cod+f"\nAdditional information:-\n {agent['input']}\n"+req_out+"\n GIVE A SUMMARY OF ABOVE INFORMATION IN POINTS. CAPTURE ALL MINUTE DETAILS. ALSO KEEP THE DETAILS OF NEW FILES CREATED OR CHANGED FILES IN DIRECTORY TEMP/ IN THE FINAL SUMMARY."})
                    print("SUMMARIZATION GENERATED:-",gemini_generated)
                    st.write(f"SUMMARIZATION GENERATED:-\n{gemini_generated}")
                    # if gemini_generated[0]=="`":
                    #     gemini_generated=gemini_generated[7:-4]
                    #     gemini_generated=json.loads(gemini_generated)
                    # else:
                    #     gemini_generated=json.loads(gemini_generated)
                    sub_note[f"output_{agent['id']}"]={f"{' , '.join(out)}":gemini_generated}
                    if ' , '.join(out)=="summary":
                        sub_note["summary"]=gemini_generated
                    
            else:
                if name=="code_runner":
                    code_to_run=""
                    lang=""
                    for i in agent['input']:
                        if agent['input'][i][:7]=="output_":
                            print("INPUT FORMAT MATCHED !!!!!!!!!!!!!!!")
                            print(sub_note)
                            try:
                                
                                if agent['input'][i][9:]=="code":
                                    agent['input'][i]=sub_note[agent['input'][i][:8]][agent['input'][i][9:]]
                                    code_to_run=agent['input'][i]
                                if agent['input'][i][9:]=="language":
                                    agent['input'][i]=sub_note[agent['input'][i][:8]][agent['input'][i][9:]]
                                    lang=agent['input'][i]
                            except:
                                
                                agent['input'][i]=sub_note[agent['input'][i][:8]]
                                code_to_run=agent['input'][i]["code"]
                                lang=agent['input'][i]["language"]
                                break
                    print("CODE TO RUN:-",code_to_run)
                    print("LANGUAGE:-",lang)
                    logs=code_runner(code_to_run,lang)
                    print("OUTPUT LOGS:-",logs)
                    st.write(f"OUTPUT LOGS:-\n")
                    st.json({"logs":logs})
                    sub_note[f"output_{agent['id']}"]={"logs":logs}
                    



                elif name=="web_search":
                    # code for web search
                    # other codes here
                    pass
                elif name=="scrapper":
                    # code for scrapper
                    # other codes here
                    pass
                elif name=="youtube_search":
                    # code for youtube search
                    # other codes here
                    pass
                elif name=="get_video_content":
                    # code for get_video_content
                    # other codes here
                    pass
                elif name=="create_text_embeddings":
                    # code for create_text_embeddings
                    # other codes here
                    pass
                elif name=="semantic_search":
                    # code for semantic_search
                    # other codes here
                    pass
                elif name=="translate":
                    # code for translate
                    # other codes here
                    pass
                    

                    
            
    notebook[f"n{sub_task['id']}"]=sub_note
            



        
     




def create_agents(question:str):
    notebook={}
    enhanced_prompt=prompt_enhancer_node(question)
    # task=Task(enhanced_prompt["task"],enhanced_prompt["inputs"],enhanced_prompt["sub_tasks"])
    
    for sub_task in enhanced_prompt["sub_tasks"]:
        print(f"PROCESSING SUBTASK {sub_task['id']}:-",sub_task['subtask_description'],"\n")
        st.write("-" * 20)
        with st.chat_message("ai"):
            st.write(f"Step: PROCESSING SUBTASK {sub_task['id']}:-\n{sub_task['subtask_description']}")
        opt_frm_prev_sub_task={}
        if sub_task["previous_knowledge"]:
            for i in sub_task['previous_knowledge']:
                inside=notebook[f"n{i}"]
                opt_frm_prev_sub_task[f"summary of previous task of {inside['subtask_description']}"]=inside["summary"]
                # for j in inside:
                #     for k in inside[j]:
                #         opt_frm_prev_sub_task[k]=inside[j][k]
        print("OUTPUTS FROM PREVIOUS SUB TASKS:-",opt_frm_prev_sub_task,"\n")
        if enhanced_prompt['inputs']:
            agent_list=rational_plan_node(sub_task['subtask_description']+"\n"+f"UPLOADED FILES BY USER:-\n{enhanced_prompt['inputs']}"+"\n"+f"ANALYSIS FROM PREVIOUS TASKS:-\n{opt_frm_prev_sub_task}")
        else:
            agent_list=rational_plan_node(sub_task['subtask_description']+"\n"+f"ANALYSIS FROM PREVIOUS TASKS:-\n{opt_frm_prev_sub_task}")
        process_subtask(agent_list,sub_task,notebook)
        st.write("-" * 20)
        st.write(f"SUBTASK {sub_task['id']} COMPLETED. FULL NOTEBOOK OUTPUTS:-")
        st.json(notebook)

        # check if notebook[f"n{sub_task["id"]}"] has all the outputs required for next subtask. if not, then reconstrct the subtask["subtask_description"] and call rational_plan_node again, describing the missing outputs

    
        



if uploaded_files:
    st.write(f"You have uploaded {len(uploaded_files)} files.")

    # Loop through each uploaded file
    for uploaded_file in uploaded_files:

        # Display file details
        st.write(f"### {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Example of reading the file content
        # Uncomment based on the file type
        if uploaded_file.type == "application/pdf":
            st.write("PDF file uploaded.")
            st.write("Currently, this app does not display PDF content. You can process it here.")
        
        elif uploaded_file.type.startswith("text/"):
            content = uploaded_file.read().decode("utf-8")
            st.text_area(f"Content of {uploaded_file.name}", content, height=200)
        
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
        
        elif uploaded_file.type.startswith("image/"):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        elif uploaded_file.type.startswith("video/"):
            st.video(uploaded_file)
        
        else:
            st.write("Unsupported file type.")

        # Optionally save the file locally
        with open(f"TEMP/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
            st.write(f"Saved file: {uploaded_file.name}")



if chat_input:
    with st.chat_message("user"):
        st.write(chat_input)
    chat_input="USER_QUERY:-\n"+chat_input
    chat_input+="\nUPLOADED FILES:-\n"
    if uploaded_files:
        for uploaded_file in uploaded_files:
            content = f"TEMP/{uploaded_file.name}"
            chat_input+=f" {content} \n"
    a=time.time()
    agents=create_agents(chat_input)   #list of agents in json list format
    b=time.time()
    st.write(f"TIME TAKEN :{(b-a):.3f} seconds")
    