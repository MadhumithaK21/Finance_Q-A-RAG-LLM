# Group4_ConvAI_RAGLLM
# Assignment 2
##AIMLCZG521

About:
A financial question-answering system using retrieval-augmented generation (RAG) and fine-tuned language models. This project includes data ingestion, document chunking, embedding storage, and evaluation scripts for financial document QA tasks. Models are fine-tuned for domain-specific accuracy and support both baseline and LoRA adapter approaches.

Group Members:
| Name                  | BITS ID                  | Contribution |
|-----------------------|--------------------------|-------------|
|   Abani Kumar Sahoo   |      2023ad05035         |  100%       |
|   Varpe Amol Devram   |      2023ad05027         |  100%       |
|   Madhumitha K        |      2023ad05077         |  100%       |
|   Thamatam Anitha     |      2023ac05985         |  100%       |
|    Madhuri S          |      2023ac05925         |  100%       |


# Financial QA Application

This application uses **Streamlit** to answer questions about the [FRTIB Financial Statements](https://catalog.data.gov/dataset/frtib-financial-statements/resource/90a07ed3-761d-4401-92e1-7b25ca73fc00) dataset.  
It supports two methods for answering questions:
- **RAG (Retrieval-Augmented Generation)**
- **Fine-tuning**

---

## **Setup Instructions**

### 1. Install Python and VS Code
- Install [Python 3.10.7](https://www.python.org/downloads/release/python-3107/)  
  > Make sure to check **"Add Python to PATH"** during installation.
- Install [Visual Studio Code](https://code.visualstudio.com/Download).

---

### 2. Clone this repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
````

---

### 3. Configure HuggingFace API Key

* Open the `.env` file in VS Code.
* Add your HuggingFace API key:

```env
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```

---

### 4. Create and activate virtual environment

```bash
python -m venv myenv
```

* **Windows**

```bash
myenv\Scripts\activate
```

* **Mac/Linux**

```bash
source myenv/bin/activate
```

---

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 6. Run the application

```bash
python -m streamlit run app.py
```

* This will open a Streamlit app in your browser at `http://localhost:8501`.


## **How to Use**

1. Select a method (**RAG** or **Fine-tuning**) from the dropdown.
2. Type your question in the input box.
3. Click **"Get answer"** and wait for the response.
4. Check if the answer is correct (**Y/N**).
5. The result will be logged automatically.


## **Data Source**

[FRTIB Financial Statements](https://catalog.data.gov/dataset/frtib-financial-statements/resource/90a07ed3-761d-4401-92e1-7b25ca73fc00)



