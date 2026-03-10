import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from .utils import (
    clean_runs, clean_generic_stat, extract_player_info, 
    clean_opposition
)
from .models import BattingStat, BowlingStat

class CricinfoScraper:
    BASE_URL = "https://stats.espncricinfo.com"
    HEADERS = {'User-Agent': 'CricinfoDataProject/1.0 (contact: arkosaha.ruet@gmail.com)'}

    def __init__(self, delay: float = 1.5):
        self.delay = delay

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetches URL and returns a BeautifulSoup object."""
        time.sleep(self.delay)
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def get_total_pages(self, soup: BeautifulSoup) -> int:
        """Extracts total pages from the pagination info."""
        try:
            # Based on the notebook: soup.find_all('td', class_='left')[3].text.split(' ')[6]
            pagination_text = soup.find_all('td', class_='left')[3].text
            parts = pagination_text.split(' ')
            if len(parts) >= 7:
                return int(parts[6].rstrip())
        except (IndexError, ValueError) as e:
            print(f"Could not determine total pages: {e}")
        return 1

    def scrape_innings_data(self, base_search_url: str, parser_func: Callable, limit_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generic method to scrape innings data (batting or bowling)."""
        all_data = []
        soup = self.get_soup(f"{self.BASE_URL}{base_search_url}&page=1")
        if not soup:
            return []

        total_pages = self.get_total_pages(soup)
        pages_to_scrape = min(total_pages, limit_pages) if limit_pages else total_pages
        
        print(f"Starting scrape: {pages_to_scrape} pages total.")

        for i in range(1, pages_to_scrape + 1):
            url = f"{self.BASE_URL}{base_search_url}&page={i}"
            if i > 1: # Already fetched page 1
                soup = self.get_soup(url)
            
            if not soup:
                break
            
            print(f"Scraping page {i}/{pages_to_scrape}...")
            
            table = soup.select_one('#ciHomeContentlhs > div.pnl650M > table:nth-child(5)')
            if table and table.tbody:
                rows = table.tbody.find_all('tr')
                for row in rows:
                    parsed_row = parser_func(row)
                    if parsed_row:
                        all_data.append(parsed_row)
            
            # Check for 'Next' link to be safe/robust
            if not soup.select_one('.PaginationLink'):
                 # Note: In some versions it might be different, but keeping the logic from NB
                 # However, we often iterate by page number anyway.
                 pass

        return all_data

    @staticmethod
    def batting_row_parser(row) -> Optional[Dict[str, Any]]:
        cols = row.find_all('td')
        if len(cols) < 12:
            return None
        
        player_name, country = extract_player_info(cols[0].text.strip())
        runs, not_out = clean_runs(cols[1].text.strip())
        
        return {
            'player_name': player_name,
            'country': country,
            'runs': runs,
            'not_out': not_out,
            'mins': int(clean_generic_stat(cols[2].text.strip())),
            'bf': int(clean_generic_stat(cols[3].text.strip())),
            'fours': int(clean_generic_stat(cols[4].text.strip())),
            'sixes': int(clean_generic_stat(cols[5].text.strip())),
            'sr': clean_generic_stat(cols[6].text.strip()),
            'inns': int(clean_generic_stat(cols[7].text.strip())),
            'opposition': clean_opposition(cols[9].text.strip()),
            'ground': cols[10].text.strip(),
            'start_date': cols[11].text.strip()
        }

    @staticmethod
    def bowling_row_parser(row) -> Optional[Dict[str, Any]]:
        cols = row.find_all('td')
        if len(cols) < 11:
            return None
        
        player_name, country = extract_player_info(cols[0].text.strip())
        
        return {
            'player_name': player_name,
            'country': country,
            'overs': clean_generic_stat(cols[1].text.strip()),
            'maidens': int(clean_generic_stat(cols[2].text.strip())),
            'runs': int(clean_generic_stat(cols[3].text.strip())),
            'wickets': int(clean_generic_stat(cols[4].text.strip())),
            'economy': clean_generic_stat(cols[5].text.strip()),
            'inns': int(clean_generic_stat(cols[6].text.strip())),
            'opposition': clean_opposition(cols[8].text.strip()),
            'ground': cols[9].text.strip(),
            'start_date': cols[10].text.strip()
        }
