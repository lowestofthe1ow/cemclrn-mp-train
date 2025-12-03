import fastapi as fa
import multiprocessing
import mysql.connector as sql
import os
import shutil as sh

from dotenv import load_dotenv
from multiprocessing import Process

from src.engines.inference import inference as inference_run
from src.scripts.train.SigNet_finetuning import finetune
from src.utils.populate_db import populate_signatures
from src.utils.transforms.transforms import TRANSFORMS_TRAIN

multiprocessing.set_start_method("spawn", force=True)

load_dotenv()
db = sql.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_DATABASE"),
)

cursor = db.cursor()


def create_tables(cursor):
    cursor.execute("DROP TABLE IF EXISTS models;")
    cursor.execute("DROP TABLE IF EXISTS signatures;")
    cursor.execute("DROP TABLE IF EXISTS users;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS signatures (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            path VARCHAR(255) NOT NULL,
            time_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
            REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            path VARCHAR(255) NOT NULL,
            time_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
            REFERENCES users(id)
        )
        """
    )

    db.commit()


def get_user_id(name):
    cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
    r = cursor.fetchone()

    if r:
        return r[0]
    else:
        return None


# Adds user if missing in database
def add_missing_user(name):
    cursor.execute("INSERT IGNORE INTO users (name) VALUES (%s)", (name,))
    db.commit()
    return get_user_id(name)


# Add new signature to database
def add_signature(user_id, path):
    cursor.execute(
        "INSERT INTO signatures (user_id, path) VALUES (%s, %s)", (user_id, path)
    )
    db.commit()


def get_15_most_recent(user_id):
    cursor.execute(
        "SELECT path FROM signatures WHERE user_id = %s ORDER BY time_added DESC LIMIT 15",
        (user_id,),
    )
    return [row[0] for row in cursor.fetchall()]


def get_model_path(user_id):
    cursor.execute(
        "SELECT path FROM models WHERE user_id = %s ORDER BY time_added DESC LIMIT 1",
        (user_id,),
    )
    row = cursor.fetchone()

    if row:
        return row[0]
    else:
        return None


def count_entries(user_id):
    cursor.execute(
        "SELECT COUNT(DISTINCT path) FROM signatures WHERE user_id = %s",
        (user_id,),
    )
    result = cursor.fetchone()
    return result[0] if result else 0


print("Starting server...")

create_tables(cursor)
populate_signatures(db, cursor)

app = fa.FastAPI()


# /register POST - Registers a new user in the database along with signature samples
@app.post("/register")
async def register(name: str, files: list[fa.UploadFile] = fa.File(...)):
    user_id = add_missing_user(name)

    folder = f"data/user_data/{name}"
    os.makedirs(folder, exist_ok=True)

    for file in files:
        path = os.path.join(folder, file.filename)
        with open(path, "wb") as f:
            sh.copyfileobj(file.file, f)
        add_signature(user_id, path)

    return {"Status": "Registered", "Signature Count": len(files)}


# /inference POST - Accepts a request body with {image, name}, returns if genuine/forged


@app.post("/inference")
async def inference(name: str, file: fa.UploadFile = fa.File(...)):
    user_id = get_user_id(name)
    if not user_id:
        return {"Error": "User not found."}

    recent = get_15_most_recent(user_id)

    # This is where model compares but idk if thats done

    file_path = os.path.join("temp", file.filename)
    with open(file_path, "wb+") as file_object:
        sh.copyfileobj(file.file, file_object)

    model_path = get_model_path(user_id)
    if model_path:
        print(f"Found model at {model_path}. Using that for inference")
    else:
        print("Defaulting to base model checkpoints/base_model.pth")
        model_path = "checkpoints/base_model.pth"

    total_dist, prediction = inference_run(
        model_path,
        file_path,
        recent,
    )

    if prediction == 1:
        prediction = "Genuine"
    else:
        prediction = "Forged"

    return {"Result": prediction, "Avg. distance": total_dist}  # placeholder


# /update POST (?) - Adds a new signature for a user.
# Check if user has > 30 (?) signature samples, fine-tune model based on most recent 30 samples
@app.post("/update")
async def update(name: str, file: fa.UploadFile = fa.File(...)):
    user_id = get_user_id(name)
    if not user_id:
        return {"Error": "User not found."}

    folder = f"data/user_data/{name}"
    os.makedirs(folder, exist_ok=True)

    num_entries = count_entries(user_id)

    _, extension = os.path.splitext(file.filename)

    # Saving file to disk
    file_path = os.path.join(folder, f"{name}_{num_entries + 1}{extension}")
    with open(file_path, "wb") as f:
        sh.copyfileobj(file.file, f)

    # Storing path in database
    add_signature(user_id, file_path)

    recent = get_15_most_recent(user_id)

    if len(recent) >= 15:
        p = Process(target=finetune, args=(user_id,))
        p.start()

    return {
        "message": "success",
    }  # placeholder, fine-tune model based on most recent 30 samples
