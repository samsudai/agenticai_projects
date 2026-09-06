import boto3
import json
import numpy as np

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

facts = [
    'The first computer was invented in the 1940s.',
    'John F. Kennedy was the 35th President of the United States.',
    'The first moon landing was in 1969.',
    'The capital of France is Paris.',
    'Earth is the third planet from the sun.',
]

question = 'Who is the president of USA?'


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def getEmbedding(input: str):
    response = client.invoke_model(
        body=json.dumps({
            "inputText": input,
        }),
        modelId='amazon.titan-embed-text-v1',
        accept='application/json',
        contentType='application/json'
    )

    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')


factsWithEmbeddings = [
    {'text': fact, 'embedding': getEmbedding(fact)}
    for fact in facts
]

questionEmbedding = getEmbedding(question)

similarities = [
    {
        'text': fact['text'],
        'similarity': cosine_similarity(fact['embedding'], questionEmbedding)
    }
    for fact in factsWithEmbeddings
]

similarities.sort(key=lambda x: x['similarity'], reverse=True)

print(f"Similarities for question: '{question}':")
for item in similarities:
    print(f"  '{item['text']}': {item['similarity']:.2f}")
