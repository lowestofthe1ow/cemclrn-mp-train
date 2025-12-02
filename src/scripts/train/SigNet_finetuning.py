import os
from datetime import datetime
import pandas as pd
import numpy as np
import mysql.connector as sql
from pathlib import Path

# SQL for user signatures
db = sql.connect(
    host="localhost",  # change if needed
    user="root",  # change if needed
    password="fujita_kotone",  # change if needed
    database="signatures",  # change if needed
)
cursor = db.cursor()
user_data_path = "data/user_data"
cedar_org_path = "data/cedar/full_org"


# Create tables
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
    """
    cursor.execute("DELETE FROM users")
    cursor.execute("DELETE FROM signatures")
    cursor.execute("ALTER TABLE users AUTO_INCREMENT = 1")
    cursor.execute("ALTER TABLE signatures AUTO_INCREMENT = 1")

    cursor.execute("DELETE FROM users")
    cursor.execute("DELETE FROM signatures")
    """
    db.commit()


# Insert user data
def insert_user_data(cursor):
    create_tables(cursor)
    base_dir = Path(user_data_path).absolute()
    if base_dir.exists():
        print(f"User data directory accessed\n")
    else:
        print("Cry")

    for user_folder in sorted(base_dir.iterdir()):
        user_name = user_folder.name

        cursor.execute("INSERT IGNORE INTO users (name) VALUES (%s)", (user_name,))
        db.commit()

        cursor.execute("SELECT id FROM users WHERE name = %s", (user_name,))

        user_id = cursor.fetchone()[0]

        image_paths = sorted(user_folder.glob("*.png"))
        image_paths.extend(sorted(user_folder.glob("*.jpg")))
        image_paths.extend(sorted(user_folder.glob("*.jpeg")))

        for image_path in image_paths:
            image_path_str = os.path.join(user_folder, image_path)
            # Time the signature was added
            timestamp = os.path.getmtime(image_path_str)
            dt_timestamp = datetime.fromtimestamp(timestamp)
            sql_timestamp = dt_timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # For checking image path and user
            # print(image_path_str + " from user " + str(user_id))
            cursor.execute(
                """
                        INSERT IGNORE INTO signatures (user_id, path, time_added)
                        VALUES (%s, %s, %s)
                        """,
                (user_id, image_path_str, sql_timestamp),
            )
        db.commit()


def makeDf(repeat, user_id):
    query = """
    SELECT path
    FROM signatures 
    WHERE user_id = %s
    ORDER BY time_added DESC 
    LIMIT 15
    """

    new_sign_df = pd.read_sql(query, db, params=(user_id,))
    new_sign_df = new_sign_df.loc[new_sign_df.index.repeat(repeat)].reset_index(
        drop=True
    )

    all_org = [
        os.path.join(cedar_org_path, f)
        for f in os.listdir(cedar_org_path)
        if os.path.isfile(os.path.join(cedar_org_path, f))
    ]
    random_orgs = np.random.choice(all_org, size=15 * repeat, replace=False)

    new_sign_df = new_sign_df.rename(columns={"path": "orig"})
    new_sign_df["not_orig"] = random_orgs

    return new_sign_df


def main():
    insert_user_data(cursor)

    # Check if correct
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    print(f"Total users in database: {total_users}")
    cursor.execute("SELECT COUNT(*) FROM signatures")
    total_signatures = cursor.fetchone()[0]
    print(f"Total signatures in database: {total_signatures}")

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    print(makeDf(5, 1))


if __name__ == "__main__":
    main()
