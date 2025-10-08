import os
import pandas as pd
import yfinance as yf


class B3DataDownloader:
    """
    A class for handling B3-listed companies and downloading their historical data
    from Yahoo Finance.
    """

    def __init__(
        self,
        companies_path: str = r"data\raw\acoes-listadas-b3.csv",
        output_path: str = r"data\processed",
    ):
        """
        Initialize the B3DataDownloader with paths for loading and saving data.

        Parameters
        ----------
        companies_path : str
            Path to the CSV file containing listed companies on B3.
        output_path : str
            Fixed path where processed data will be exported.
        """
        self.companies_path = companies_path
        self.output_path = output_path
        self.listed_companies = None

    # ---------------------------------------------------------------------- #
    #                            LOAD COMPANIES                              #
    # ---------------------------------------------------------------------- #
    def load_listed_companies(self) -> pd.DataFrame | None:
        """
        Load a list of companies listed on B3 from a CSV file.
        The CSV file must contain a 'Ticker' column.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with listed companies or None if loading fails.
        """
        try:
            df = pd.read_csv(self.companies_path)
            if "Ticker" not in df.columns:
                raise ValueError("The CSV file must contain a 'Ticker' column.")
            self.listed_companies = df
            return df

        except FileNotFoundError:
            print(f"❌ File not found: {self.companies_path}")
            return None

        except Exception as e:
            print(f"⚠️ Failed to load listed companies: {e}")
            return None

    # ---------------------------------------------------------------------- #
    #                             CHECK TICKER                               #
    # ---------------------------------------------------------------------- #
    def check_ticker(self) -> str | None:
        """
        Continuously prompt the user to enter a ticker symbol until a valid one is provided.

        Returns
        -------
        str
            Valid ticker symbol with '.SA' suffix (uppercase).
        """
        if self.listed_companies is None:
            print("⚠️ Listed companies not loaded. Run 'load_listed_companies()' first.")
            return None

        while True:
            try:
                ticker = input("Enter ticker symbol: ").strip().upper()

                if not ticker:
                    print("⚠️ You didn't type anything. Please try again.")
                    continue

                if ticker in self.listed_companies["Ticker"].values:
                    print(f"✅ Ticker '{ticker}' found.")
                    return ticker + ".SA"

                print(f"❌ '{ticker}' not found in the list. Please try again.")

            except Exception as e:
                print(f"⚠️ Unexpected error ({type(e).__name__}): {e}")
                print("Please try again...")

    # ---------------------------------------------------------------------- #
    #                         DOWNLOAD HISTORICAL DATA                       #
    # ---------------------------------------------------------------------- #
    def download_full_history(self, ticker: str) -> pd.DataFrame | None:
        """
        Download the full available daily historical data for a given ticker from Yahoo Finance.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., 'PETR4.SA').

        Returns
        -------
        pd.DataFrame or None
            Historical data for the ticker, or None if download fails.
        """
        try:
            print(f"⬇️ Downloading full historical daily data for {ticker}...")
            data = yf.download(
                tickers=ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=True,
                threads=True,
                group_by="column",
                timeout=20,
            )

            if data.empty:
                print(f"⚠️ No historical data found for '{ticker}'.")
                return None

            print(f"✅ Successfully downloaded {len(data)} records for {ticker}.")
            print(
                f"📆 From {data.index.min().strftime('%Y-%m-%d')} "
                f"to {data.index.max().strftime('%Y-%m-%d')}"
            )
            return data

        except Exception as e:
            print(f"❌ Failed to download data for '{ticker}': {e}")
            return None

    # ---------------------------------------------------------------------- #
    #                              EXPORT DATA                               #
    # ---------------------------------------------------------------------- #
    def export_data(self, data: pd.DataFrame, ticker: str) -> None:
        """
        Export downloaded historical data to a CSV file.

        Parameters
        ----------
        data : pd.DataFrame
            The data to export.
        ticker : str
            Ticker symbol (used for file naming).
        """
        try:
            os.makedirs(self.output_path, exist_ok=True)  # cria a pasta processed
            filename = "df.csv"
            filepath = os.path.join(self.output_path, filename)
            data.columns = data.columns.droplevel(1) 
            data.to_csv(filepath, index=True)
            print(f"💾 Data saved successfully at: {filepath}")

        except Exception as e:
            print(f"❌ Failed to export data: {e}")

    # ---------------------------------------------------------------------- #
    #                              MAIN RUNNER                               #
    # ---------------------------------------------------------------------- #
    def run(self) -> pd.DataFrame | None:
        """
        Execute the complete workflow:
        - Load companies
        - Ask for ticker
        - Download data
        - Export CSV

        Returns
        -------
        pd.DataFrame or None
            The downloaded data.
        """
        self.load_listed_companies()
        if self.listed_companies is None:
            return None

        ticker = self.check_ticker()
        if ticker is None:
            return None

        data = self.download_full_history(ticker)
        if data is None:
            return None

        self.export_data(data, ticker)
        return data

# execution example
# if __name__ == "__main__":
#     downloader = B3DataDownloader()
#     df = downloader.run()