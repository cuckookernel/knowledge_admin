# Knowledge Admin Package

A set of tools for knowledge transformation and administration supported by AI models of various kinds.



### Installation 

Important: Create a venv first! For example:

```bash
python 3.10 -m venv ~/venv-k-adm
source ~/venv-k-adm/bin/activate.sh
```

After activating your venv, run:

```bash
pip install git+https://github.com/cuckookernel/knowledge_admin.git
```

For document to audio conversion (e.g. txt/pdf -> mp3/wav) run:
```
doc_2_audio --help
```


For text summarization, install `langchain` and `langchain-anthropic` and then run:
```
summarize_text --help
```
