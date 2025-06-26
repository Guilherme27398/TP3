import textwrap
import markdown
import pandas as pd
from flask import Flask, request, render_template
from chromadb import PersistentClient
from embedding import SentenceTransformerEmbeddingFunction
import ollama

app = Flask(__name__)


def initialize_components():
    global embedding_func, db_client, collection
    embedding_func = SentenceTransformerEmbeddingFunction()
    embedding_func.initialize_model()

    db_client = PersistentClient(path="UroBot_database")
    collection = db_client.get_collection(name="UroBot_v1.0", embedding_function=embedding_func)

initialize_components()


def convert_markdown_to_html_or_text(input_text):
    lines = input_text.strip().split('\n')
    output = ""
    inside_table = False
    table_started = False
    alignments = []
    html_table = ''

    for i, line in enumerate(lines):
        if (not table_started and '|' in line and i + 1 < len(lines) and
                '|' in lines[i + 1] and all(c in '|:- ' for c in lines[i + 1].strip())):
            if not inside_table:
                if output.strip():
                    output += "<p>" + output.strip() + "</p>\n"
                output += '<table>\n'
                inside_table = True
                table_started = True
                html_table = '  <tr>\n'
            continue
        elif table_started and line.strip() == "":
            output += html_table + '</table>\n'
            inside_table = False
            table_started = False
            alignments = []
            continue

        if inside_table:
            if table_started and all(c in '|:- ' for c in line.strip()):
                alignments = [
                    'center' if cell.strip().startswith(':') and cell.strip().endswith(':') else
                    'right' if cell.strip().endswith(':') else
                    'left' for cell in line.strip('|').split('|')
                ]
                table_started = False
                continue
            cells = line.strip('|').split('|')
            cell_tag = 'th' if table_started else 'td'
            for idx, cell in enumerate(cells):
                align_style = f' style="text-align: {alignments[idx]};"' if alignments else ''
                html_table += f'    <{cell_tag}{align_style}>{cell.strip()}</{cell_tag}>\n'
            html_table += '  </tr>\n'

        else:
            if output.strip():
                output += line + "\n"
            else:
                output = line + "\n"

    if inside_table:
        output += html_table + '</table>\n'

    return output.strip()


def process_query(query):
    query_results = collection.query(query_texts=[query], n_results=9)
    context = ""
    documents = []

    for i, item in enumerate(query_results["documents"][0]):
        id = query_results["ids"][0][i]
        context += f"\nDocument ID {id[2:]}:\n{item}\n"
        if query_results["metadatas"][0][i]["paragraph_type"] == "table":
            df = pd.read_csv(query_results["metadatas"][0][i]["dataframe"]).to_html(index=False)
            documents.append(f"Document ID {id[2:]}:\n \n{df} \n")
        else:
            documents.append(f"Document ID {id[2:]}:\n \n{convert_markdown_to_html_or_text(item)} \n")

    full_prompt = (
        "You are a helpful and understanding urologist answering questions to the patient.\n"
        "Use full sentences and answer in a human-like tone. After the answer, ask if you can help further.\n"
        "Base your answer on the following context:\n"
        "---\n"
        f"{context}\n"
        "---\n"
        "If the context does not provide information on the question, respond with:\n"
        "'Sorry my knowledge base does not include information on that topic.'\n"
        "Ensure your answer is annotated with the Document IDs of the context used. "
        "Use the format: (Document ID 'number')."
    )

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response['message']['content'], documents


@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    query = None
    documents = None
    if request.method == 'POST':
        query = request.form['query']
        answer, documents = process_query(query)

    return render_template('index.html', answer=answer, query=query, documents=documents)


if __name__ == '__main__':
    app.run(debug=True)
