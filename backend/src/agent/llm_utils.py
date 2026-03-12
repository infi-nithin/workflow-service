from langchain_aws import ChatBedrock
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm_client() -> ChatBedrock:
    """Get or create the LLM client using AWS Bedrock.
    
    Returns:
        ChatBedrock instance configured with AWS credentials
    """
    return ChatBedrock(
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        region_name=os.getenv("AWS-REGION"),
        model_id=os.getenv("BEDROCK_MODEL_ID"),
        model_kwargs={
            "temperature": 0.3
        }
    )