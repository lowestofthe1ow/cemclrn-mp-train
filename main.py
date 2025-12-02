# ==========================================================================================================================================================
# Stuff Left: Manage file paths and name inputs in routes
# ==========================================================================================================================================================
import mysql.connector as sql
import fastapi as fa
import os
import shutil as sh

from src.utils.transforms.transforms import TRANSFORMS_TRAIN


db = sql.connect(
    host="localhost",  # change if needed
    user="root",  # change if needed
    password="fujita_kotone",  # change if needed
    database="signatures",  # change if needed
)

cursor = db.cursor()


def create_tables(cursor):
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


# Get 30 most recent signatures
def get_30_most_recent(user_id):
    cursor.execute(
        "SELECT path FROM signatures WHERE user_id = %s ORDER BY time_added DESC LIMIT 30",
        (user_id,),
    )
    return [row[0] for row in cursor.fetchall()]


app = fa.FastAPI()


# /register POST - Registers a new user in the database along with signature samples
@app.post("/register")
async def register(name: str, files: list[fa.UploadFile] = fa.File(...)):
    user_id = add_missing_user(name)

    folder = f"data/{name}"
    os.makedirs(folder, exist_ok=True)

    for file in files:
        path = os.path.join(folder, file.filename)
        with open(path, "wb") as f:
            sh.copyfileobj(file.file, f)
        add_signature(user_id, path)

    return {"Status": "Registered", "Signature Count": len(files)}


# /inference POST - Accepts a request body with {image, name}, returns if genuine/forged


@app.post("/inference")
async def inference(name: str, new_signature: fa.UploadFile = fa.File(...)):
    user_id = get_user_id(name)
    if not user_id:
        return {"Error": "User not found."}

    recent = get_30_most_recent(user_id)

    # This is where model compares but idk if thats done

    return {"Result": "Genuine", "Confidence": 0.91}  # placeholder


# /update POST (?) - Adds a new signature for a user.
# Check if user has > 30 (?) signature samples, fine-tune model based on most recent 30 samples


@app.post("/update")
async def update(name: str, file: fa.UploadFile = fa.File(...)):
    user_id = get_user_id(name)
    if not user_id:
        return {"Error": "User not found."}

    add_signature(user_id, file)  # havent finished this, needs to be path not file

    recent = get_30_most_recent(user_id)

    if len(recent) > 30:
        return {
            "Status": "Registered",
            "Signature Count": 67,
        }  # placeholder, fine-tune model based on most recent 30 samples


def main():
    print("Hello from cemclrn-mp-train!")
    create_tables(cursor)


if __name__ == "__main__":
    main()
