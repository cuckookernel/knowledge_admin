"""Summarize a long text by chunks using an LLM

Example Usages:

summarize_text some_file.txt -r 0.4  # summarize to approx. 40%% of the length

summarize_text some_file.txt -m gpt-3.5-turbo  # summarize to approx. 40%% of the length

IMPORTANT: you need to set the appropiate LLM API beforehand.
Example:

For the Anthropic models (default)
export ANTHROPIC_API_KEY=.....

If using GPT model
export OPENAI_API_KEY=.....

"""

import argparse
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


from k_admin.text_util import count_words


def main():
    args = get_cli_args()

    in_file = Path(args.input_file)
    text = in_file.read_text(encoding=args.encoding)
    print(f'INFO: total length of input text: {len(text)}')

    chat_model = get_llm(args.llm)

    summary_text = summarize_by_chunks(chat_model=chat_model,
                                       text=text,
                                       ratio=args.ratio,
                                       instruction_lang=args.instruction_lang,
                                       chunk_size=args.chunk_size)

    out_file_path = in_file.with_suffix('.summary.txt')
    out_file_path.write_text(summary_text)
    print(f'\nINFO: summary written to: {str(out_file_path)}, length is {len(summary_text)}')


def get_cli_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('input_file', help='path to input text, should be of type .txt')
    arg_parser.add_argument('-e', '--encoding', help="input text file's encoding",
                            default="utf8")
    arg_parser.add_argument('-m', '--llm',
                            help='External LLM model used via langchain, e.g. claude-3-sonnet... '
                                 'or gpt-...',
                            default="claude-3-sonnet-20240229")

    arg_parser.add_argument('-r', '--ratio',
                            help='Summarization ratio as a real number (with decimals) between 0.0 '
                                 'and 1.0, e.g. use 0.3 for a summary that whose length is 30%% '
                                 'the length of the original text',
                            type=float,
                            default=0.3)

    arg_parser.add_argument('-c', '--chunk-size',
                            help='The max size of each chunk in characters',
                            type=int, default=3000)

    arg_parser.add_argument('-l', '--instruction-lang',
                            help='language for internal instruction passed to llm model',
                            default="en")

    return arg_parser.parse_args()


def get_llm(model_name: str) -> BaseChatModel:
    if model_name.startswith("claude-"):
        return ChatAnthropic(model=model_name)
    elif model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=0)
    else:
        raise ValueError("Only OpenAI's and claude models from Anthropic supported at this time")


def summarize_by_chunks(*, chat_model: BaseChatModel, text: str, ratio: float,
                        instruction_lang: str, chunk_size: int) -> str:

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    summary_pieces = []
    for i, piece in enumerate(splitter.split_text(text)):
        print(f"\n\n################################ - \nPiece {i} (len: {len(piece)} ):\n{piece}")

        summarized_piece = summarize(chat_model=chat_model, text=piece,
                                     ratio=ratio,
                                     instruction_lang=instruction_lang)
        print(f"\n\n###### \nSummary of piece {i} (len: {len(summarized_piece)}):\n"
              f"{summarized_piece}")
        summary_pieces.append(summarized_piece)

    return "\n".join(summary_pieces)


def summarize(chat_model: BaseChatModel, text: str, ratio: float = 0.30,
              instruction_lang: str = "en") -> str:

    if instruction_lang == "en":
        system_msg = ("You are a helpful assistant that summarizes text provided by the user, "
                      "without adding any content not already contained in the user's original "
                      "text, and also without changing the basic style")

        human_msg = """Please produce a summary of the following text in at most {n_words_out} 
          words, avoid introductory phrases such that start with "the text discusses/treats/etc.": 
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


def _test():
    # %%
    chat_model = ChatAnthropic(model="claude-3-sonnet-20240229")

    text = """
    Dilema ético de si debería o no limosna a una persona en la calle.    

Analizando el dilema desde la ética de la virtud

Para analizar el problema desde la ética de la virtud, debo considerar esencialmente la pregunta de 
cómo dar o no limosna, sobretodo si lo convierto en un hábito, me ayudaría a cultivar virtudes como la generosidad y la prudencia, y contribuirían a formar la persona que quiero ser.

Por un lado, dar limosna me haría ver como una persona que cultiva la compasión y generosidad.  
También debería considerar, desde el punto de vista de mi sabiduría práctica, si el dar limosna causará un beneficio real a largo plazo y alineado con mis intenciones que en últimas son buenas.

Sin embargo también hay que considerar la virtud de la justicia. En particular, debería preguntarme: es justo darle dinero “gratis” a una persona mientras otros se matan trabajando y aportando a la sociedad para recibir el mismo dinero? No será que al dar dinero estoy apoyando las estructuras sociales que son las causas raíces de la pobreza y la desigualdad. 

Mi conclusión bajo el enfoque virtuoso es que si bien darle dinero a un limosnero es un acto de gentileza, sería más “virtuoso” emplear ese dinero, tiempo  y esfuerzo hacia un cambio sistémico que dirigido contra las causas de la pobreza.
"""
    print("Text length: ", count_words(text), "palabras")

    summary = summarize(chat_model, text, 0.75)

    print("Summary length:", count_words(summary), "palabras")

    print(summary)
    # %%


if __name__ == "__main__":
    main()
