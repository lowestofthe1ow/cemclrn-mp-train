<div align="center">

<h1>HINA: Hybrid Writer-Independent-Dependent Network for Signature Authentication</h1>

<p><a href="https://drive.google.com/file/d/1vsZlQ9A45vEjtA5R-mIu-RvR0HxJFu_s/view?usp=sharing">Report paper</a></p>

</div>

> [!NOTE]
> **This repository contains code for the Python web server** that hosts a basic API for inference, user registration, and fine-tuning. It makes use of The Flutter mobile application can be found [here](https://github.com/lowestofthe1ow/cemclrn-mp). The paper's LaTeX source can be found [here](https://github.com/lowestofthe1ow/cemclrn-mp-report).

We propose a hybrid approach to signature verification in an attempt to make the best of the advantages of writer-dependent and writer-independent approaches. Our system utilizes a base model trained for general inference. When users are first enrolled into the system, any queries will use this base model until the system has collected enough signatures, after which it begins to train a fine-tuned copy of the model using the collected data. This fine-tuned model will then be used for further inference and will be re-trained whenever the user uploads signatures.

## Running the server

1. Ensure you have access to a running MySQL server. 
2. Create a `.env` file with your SQL database credentials following this structure:

```
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=fujita_kotone
DB_DATABASE=signatures
```

3. Start the server with `uvicorn main:app --reload --host 0.0.0.0 --port 8000`. You should now be able to access the API.

## API endpoints

`POST /register`: Registers a new user in the database along with signature samples.
- `name`: The name of the user
- `files`: The signature image file/s

`POST /inference`: Performs inference on a query image
- `name`: The name of the user
- `file`: The signature image file

`POST /update`: Performs inference on a query image
- `name`: The name of the user
- `file`: The signature image file
