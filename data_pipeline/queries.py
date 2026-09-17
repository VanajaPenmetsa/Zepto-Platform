import sqlite3
import pandas as pd

def run():
    conn = sqlite3.connect("data_pipeline/zepto_catalog.db")

    sql_join = """
        SELECT b.title, c.category_name, b.price_gbp, b.price_inr, b.rating 
        FROM books b 
        JOIN categories c ON b.category_id = c.category_id 
        WHERE b.rating = 5 
        ORDER BY b.price_gbp DESC 
        LIMIT 10;
    """
    sql_df = pd.read_sql_query(sql_join, conn)

    books_df = pd.read_sql_query("SELECT * FROM books", conn)
    categories_df = pd.read_sql_query("SELECT * FROM categories", conn)

    pandas_df = (
        pd.merge(books_df, categories_df, on="category_id")
        .query("rating == 5")
        .sort_values(by="price_gbp", ascending=False)
        .head(10)[["title", "category_name", "price_gbp", "price_inr", "rating"]]
        .reset_index(drop=True)
    )

    print("SQL vs Pandas Match:", sql_df.equals(pandas_df))
    conn.close()

if __name__ == "__main__":
    run()