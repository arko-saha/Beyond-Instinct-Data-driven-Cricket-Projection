import os
import pandas as pd
import argparse
from src.scraper.engine import CricinfoScraper

def main():
    parser = argparse.ArgumentParser(description="ESPN Cricinfo Scraper CLI")
    parser.add_argument("--type", choices=["batting", "bowling"], required=True, help="Type of stats to scrape")
    parser.add_argument("--limit-pages", type=int, help="Limit number of pages to scrape for testing")
    parser.add_argument("--output", type=str, help="Output CSV path")
    
    args = parser.parse_args()
    
    scraper = CricinfoScraper()
    
    if args.type == "batting":
        base_url_path = "/ci/engine/stats/index.html?class=3;spanmin1=1+Jan+2022;spanval1=span;template=results;type=batting;view=innings"
        data = scraper.scrape_innings_data(base_url_path, scraper.batting_row_parser, limit_pages=args.limit_pages)
        default_output = "data/batters_new.csv"
    else:
        base_url_path = "/ci/engine/stats/index.html?class=3;spanmin1=1+Jan+2022;spanval1=span;template=results;type=bowling;view=innings"
        data = scraper.scrape_innings_data(base_url_path, scraper.bowling_row_parser, limit_pages=args.limit_pages)
        default_output = "data/bowlers_new.csv"

    output_path = args.output or default_output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}. Total rows: {len(df)}")

if __name__ == "__main__":
    main()
