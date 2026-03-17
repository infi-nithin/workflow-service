from langchain_aws import ChatBedrock
from config.config import config


def get_llm_client() -> ChatBedrock:
    return ChatBedrock(
        model_id=config.aws.bedrock_model_id,
        aws_secret_access_key=config.aws.secret_access_key,
        aws_access_key_id=config.aws.access_key_id,
        aws_session_token=config.aws.session_token,
        region_name=config.aws.region,
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    )
