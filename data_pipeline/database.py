import sqlite3
import pandas as pd
from scraper import scrape_books

def build_database():
    df = scrape_books(5)
    conn = sqlite3.connect("data_pipeline/zepto_catalog.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)

    for cat in df["category_name"].unique():
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()

    cat_map = pd.read_sql("SELECT * FROM categories", conn).set_index("category_name")["category_id"]
    df["category_id"] = df["category_name"].map(cat_map)

    for _, row in df.iterrows():
        cursor.execute("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (row["title"], row["price_gbp"], row["price_inr"], row["rating"], int(row["in_stock"]), row["category_id"]))

    conn.commit()
    conn.close()
    print("Database built successfully!")

if __name__ == "__main__":
    build_database()