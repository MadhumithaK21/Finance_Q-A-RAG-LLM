Data Source: https://catalog.data.gov/dataset/frtib-financial-statements/resource/90a07ed3-761d-4401-92e1-7b25ca73fc00

Install Python: 3.10.7 and VS code

Open financial_qa with vs code

Open .env file -> copy the huggingface api key

Open cmd terminal in vscode

	a. create a virtual environment --> `python -m venv myenv`

	b. activate the environment --> `myenv\scripts\activate`

	c. install dependencies --> `pip install -r requirements.txt`

	d. run the application --> `python -m streamlit run app.py`  -> this will redirect to app

	e. Select the method (RAG/Fine tuning) -> Type the question -> Click Get answer(wait..) -> Check if correct (Y/N) -> log it