
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector

##################################################
# LLM model
##################################################
model = OllamaLLM(model="llama3.2:1b", temperature=0)

##################################################
# 1. Simple Prompt
# ---
# This is the most basic way to interact with an LLM: using a plain string with .format().
# It is quick and easy, but not reusable or scalable for more complex tasks.
##################################################
# A basic prompt string for reference
simple_prompt = "Translate the following English word to French: {word}"
formatted_simple = simple_prompt.format(word="cat")
print("Simple prompt example:")
print(formatted_simple)
print("LLM output:")
print(model.invoke(formatted_simple))
print("\n" + "-"*50 + "\n")

##################################################
# 2. PromptTemplate
# ---
# PromptTemplate is better than a simple prompt string because:
# - It is reusable and parameterized for different variables.
# - It integrates with LangChain's prompt management and validation.
# - It is easier to maintain and extend for more complex prompt logic.
##################################################
prompt_template = PromptTemplate(
    input_variables=["word"],
    template="Translate the following English word to French: {word}"
)
formatted_template = prompt_template.format(word="dog")
print("PromptTemplate example:")
print(formatted_template)
print("LLM output:")
print(model.invoke(formatted_template))
print("\n" + "-"*50 + "\n")

##################################################
# 3. FewShotPromptTemplate
# ---
# FewShotPromptTemplate is better than PromptTemplate because:
# - It allows you to provide multiple examples (few-shot learning) to guide the LLM.
# - This improves the model's accuracy and consistency for tasks like translation, classification, etc.
# - It is especially useful for tasks where context/examples matter.
##################################################
# Few-shot examples for translation
examples = [
    {"word": "cat", "translation": "chat"},
    {"word": "dog", "translation": "chien"},
    {"word": "bird", "translation": "oiseau"},
]

example_prompt = PromptTemplate(
    input_variables=["word", "translation"],
    template="English: {word}\nFrench: {translation}"
)

prefix = (
    "Translate the following English word to French.\n"
    "Only output the French word, nothing else.\n"
    "Examples:\n"
)
suffix = "English: {input_word}\nFrench:"

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input_word"],
    example_separator="\n\n"
)
formatted_few_shot = few_shot_prompt.format(input_word="apple")
print("FewShotPromptTemplate example:")
print(formatted_few_shot)
print("LLM output:")
print(model.invoke(formatted_few_shot))
print("\n" + "-"*50 + "\n")

##################################################
# 4. LengthBasedExampleSelector with FewShotPromptTemplate
# ---
# LengthBasedExampleSelector is better than static few-shot examples because:
# - It dynamically selects which examples to include based on the input length and prompt size constraints.
# - This helps avoid exceeding model context limits and keeps prompts efficient.
# - It is ideal for production scenarios where prompt length must be managed automatically.
##################################################
# This selector limits the number of examples based on prompt length
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50,  # Lower for demonstration; adjust as needed
)

dynamic_few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input_word"],
    example_separator="\n\n"
)
formatted_dynamic_few_shot = dynamic_few_shot_prompt.format(input_word="elephant")
print("FewShotPromptTemplate with LengthBasedExampleSelector example:")
print(formatted_dynamic_few_shot)
print("LLM output:")
print(model.invoke(formatted_dynamic_few_shot))
print("\n" + "-"*50 + "\n")
