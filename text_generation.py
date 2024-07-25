import json
import os
import sys

import boto3
import botocore

boto3_bedrock = boto3.client("bedrock-runtime")
prompt_data = """
Command: Write an email from Bob, Customer Service Manager, to the customer "John Doe" 
who provided negative feedback on the service provided by our customer support 
engineer"""
body = json.dumps(
    {
        "inputText": prompt_data,
        "textGenerationConfig": {"topP": 0.95, "temperature": 0.1},
    }
)
modelId = "amazon.titan-tg1-large"
accept = "application/json"
contentType = "application/json"
outputText = "\n"
try:

    response = boto3_bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    outputText = response_body.get("results")[0].get("outputText")

except botocore.exceptions.ClientError as error:

    if error.response["Error"]["Code"] == "AccessDeniedException":
        print(
            f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
        )

    else:
        raise error
email = outputText[outputText.index("\n") + 1 :]
print(email)
output = []
try:

    response = boto3_bedrock.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    stream = response.get("body")

    i = 1
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_obj = json.loads(chunk.get("bytes").decode())
                text = chunk_obj["outputText"]
                output.append(text)
                print(f"\t\t\x1b[31m**Chunk {i}**\x1b[0m\n{text}\n")
                i += 1

except botocore.exceptions.ClientError as error:

    if error.response["Error"]["Code"] == "AccessDeniedException":
        print(
            f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
        )

    else:
        raise error
