from langchain_core.prompts import PromptTemplate

prompts = PromptTemplate(
    template=""""
    You are a recommendation system which will recommend new and unique youtube content based on the given context.
    Generate about 5 to 10 different content types.

    The following are summarized description of top video which the user is interested in : \n 
    {top_video}
    
    Be Creative. Generate new idea. Only include the description of your recommended content in your answer.
    
    Provide your recommend content below:
    """,
    input_variables=["top_video"]
)