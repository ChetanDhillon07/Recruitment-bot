# utils/functions.py
from langchain_core.documents import Document

def create_document(full_name, years_of_exp, desired_positions, tech_stack):
    text = (
        f"My name is {full_name}, I have {years_of_exp} years of experience, "
        f"the position(s) I want to apply for is {desired_positions} and "
        f"my tech stack includes {tech_stack}."
    )
    doc = Document(
        page_content=text,
        metadata={
            "type": "candidate_profile",
            "full_name": full_name,
            "years_of_exp": years_of_exp,
            "desired_positions": desired_positions,
            "tech_stack": tech_stack,
        },
    )
    return [doc]  # list
