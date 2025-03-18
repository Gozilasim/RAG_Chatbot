

def getPrompt(format_choose):

    qa_system_prompt = ""

    if format_choose == "Tabular":
        qa_system_prompt = """

        You are an expert assistant.
        Respond solely based on the provided context: {context}    
        Present a detailed response in a table format .  
        Do not include any text outside of the Table.
        Response: 
        """
    elif format_choose == "List":
        qa_system_prompt = """

        You are an expert assistant.
        Respond solely based on the provided context: {context}    
        Present a detailed response in a list format.  
        Do not include any text outside of the list.
        Response: 
        """
    elif format_choose == "Paragraph":
         qa_system_prompt = """

        You are an expert assistant.
        Respond solely based on the provided context: {context}    
        Present a detailed response in a paragraph format .  
        Response: 
        """

    elif format_choose == "Graph":
        qa_system_prompt = """

        You are an expert assistant.
        Answer the question based ONLY on the following context : {context}
        Please provide a detailed and accurate answer in a paragraph format and professional tone.
        Response: 
        """


    return qa_system_prompt