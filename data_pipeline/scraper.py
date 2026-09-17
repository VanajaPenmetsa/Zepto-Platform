import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

FIXED_GBP_TO_INR = 105.50
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape_books(num_pages: int = 5) -> pd.DataFrame:
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"
    books_data = []

    for page in range(1, num_pages + 1):
        response = requests.get(base_url.format(page))
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article", class_="product_pod")

        for article in articles:
            title = article.h3.a["title"]
            price_text = article.find("p", class_="price_color").text
            
            # Extract numbers and decimal points only (removes Â, £, etc.)
            clean_price = re.sub(r"[^\d.]", "", price_text)
            price_gbp = float(clean_price)

            rating_class = article.find("p", class_="star-rating")["class"]
            rating_str = [c for c in rating_class if c != "star-rating"][0]
            rating = RATING_MAP.get(rating_str, None)

            avail_text = article.find("p", class_="instock availability").text.strip()
            in_stock = "In stock" in avail_text

            books_data.append({
                "title": title,
                "price_gbp": price_gbp,
                "rating": rating,
                "in_stock": in_stock,
                "category_name": "General Catalogue"
            })

    df = pd.DataFrame(books_data)
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    return df

if __name__ == "__main__":
    df = scrape_books(5)
    print(f"Scraped {len(df)} records successfully.")