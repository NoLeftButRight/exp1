def get_response(sentence, lang_type):
    client = OpenAI(
        api_key="sk-deT2kkPmlSIOB9zoVT1y1wyrY7oALga36yVmkAGZAhG0E496",
        base_url="https://api.chatanywhere.tech",  # 填写base_url
    )
    content_system_en = (
        'You are an assistant skilled in information extraction. Your goal is to extract named entities from the given text and classify the entity types into one of the following categories：['
        + word_types_cn_str
        + ']. The length of the entity should be as short as possible. There is no need to provide an analysis process. Regardless of any circumstances, only return the specified data format and do not return any other content. The specified return data format is as follows：[{"entity":"Lily","type":"Person"},{"entity":"German","type":"Miscellaneous"},{"entity":"Commission","type":"Organization"},{"entity":"Germany","type":"Location"}].'
    )
    content_user_en = (
        'Please identify which entities are present in the given sentence and determine the type of each entity. The entity word is only from the sentence. Entity classes can only be selected from the given types. The sentence is：'
        + sentence
        + '. The specified entity types are [' + word_types_cn_str + '].'
    )
    if lang_type == "cn":
        content_system = content_system_cn
        content_user = content_user_cn
    elif lang_type == "en":
        content_system = content_system_en
        content_user = content_user_en

    completion = client.chat.completions.create(
        model="qwen2:7b",
        messages=[
            {'role': 'system', 'content': content_system},
            {'role': 'user', 'content': content_user}
        ]
    )
    return completion.model_dump_json()