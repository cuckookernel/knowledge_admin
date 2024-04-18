
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

Vector = list

# %%


def classify_linear(x: Vector[float]) -> str:
    P = x[0]
    H = x[1]

    score = 10 * H - 45 * P - 150
    if score > 0:
        return "Gato"
    else:
        return "Perro"


def main():

    # %%
    chat_model = ChatAnthropic(model="claude-3-sonnet-20240229")

    text = """
    Dilema ético de si debería o no limosna a una persona en la calle.    
    
Analizando el dilema desde la ética de la virtud

Para analizar el problema desde la ética de la virtud, debo considerar esencialmente la pregunta de cómo dar o no limosna, sobretodo si lo convierto en un hábito, me ayudaría a cultivar virtudes como la generosidad y la prudencia, y contribuirían a formar la persona que quiero ser.

Por un lado, dar limosna me haría ver como una persona que cultiva la compasión y generosidad.  También debería considerar, desde el punto de vista de mi sabiduría práctica, si el dar limosna causará un beneficio real a largo plazo y alineado con mis intenciones que en últimas son buenas.

Sin embargo también hay que considerar la virtud de la justicia. En particular, debería preguntarme: es justo darle dinero “gratis” a una persona mientras otros se matan trabajando y aportando a la sociedad para recibir el mismo dinero? No será que al dar dinero estoy apoyando las estructuras sociales que son las causas raíces de la pobreza y la desigualdad. 

Mi conclusión bajo el enfoque virtuoso es que si bien darle dinero a un limosnero es un acto de gentileza, sería más “virtuoso” emplear ese dinero, tiempo  y esfuerzo hacia un cambio sistémico que dirigido contra las causas de la pobreza.
"""
    print("Text length: ", count_words(text), "palabras")

    summary = summarize(chat_model, text, 0.75)

    print("Summary length:", count_words(summary), "palabras")

    print(summary)
    # %%


def count_words(text: str) -> int:
    parts = re.split(r"\b", text)

    return sum(1 if not is_space(a_str) else 0
               for a_str in parts)
# %%


def is_space(a_str: str) -> bool:
    return a_str == ' ' or re.match(r'[,.:?()!]?\s+', a_str)
# %%


def summarize(chat_model: BaseChatModel, text: str, ratio: float = 0.30,
              instruction_lang: str = "en") -> str:

    if instruction_lang == "en":
        system_msg = ("You are a helpful assistant that summarizes text provided by the user, "
                      "without adding any content not already contained in the user's original "
                      "text, and also without changing the basic style")

        human_msg = """Please produce a summary of the following text in at most {n_words_out} 
          words: 
        ```
        {text}
        ```
        """
    elif instruction_lang == "es":
        system_msg = (
            "Eres un asistente que resume el texto provisto por tu usuario sin agregar ningún "
            "contenido que no se encuentra en el texto original.")

        human_msg = """Por favor produce un resumen del siguiente texto de no más de {n_words_out} 
         palabras: 
                ```
                {text}
                ```
                """
    else:
        raise ValueError(f"Invalid instruction_lang: `{instruction_lang}`, valid values "
                         f"for now are `en` or `es`.")

    prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("human", human_msg)])

    n_words_in = count_words(text)

    chain = prompt | chat_model
    result = chain.invoke({
        "text": text,
        "n_words_out": int(n_words_in) * ratio
    })

    return result.content
    # %%
