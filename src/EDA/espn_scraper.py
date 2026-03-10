import pandas as pd
import requests

class EspnCricinfoScraper:
    """Scrapes match results and historical data from ESPN Cricinfo."""

    def fetch_team_results(self, url):
        """
        Fetches match results from a given ESPN Cricinfo team results URL.
        Drops unnecessary columns as per analysis requirements.
        """
        try:
            # read_html returns a list of dataframes
            all_tables = pd.read_html(url)
            
            # The match results table is usually the 4th table in this template
            if len(all_tables) < 4:
                raise ValueError(f"Expected at least 4 tables, but found {len(all_tables)} at {url}")
                
            match_results_df = all_tables[3]
            
            # Drop unnecessary columns if they exist
            cols_to_drop = ['BR', 'Unnamed: 5', 'Unnamed: 9']
            existing_cols_to_drop = [c for c in cols_to_drop if c in match_results_df.columns]
            
            if existing_cols_to_drop:
                match_results_df = match_results_df.drop(columns=existing_cols_to_drop)
                
            return match_results_df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return pd.DataFrame()
