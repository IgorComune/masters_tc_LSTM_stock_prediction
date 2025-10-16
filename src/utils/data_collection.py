import os
import numpy as np
from typing import Optional
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Display and plotting setup
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
plt.rcParams["figure.figsize"] = (10, 5)

# Default paths
COMPANIES_PATH = r"data/raw/acoes-listadas-b3.csv"
OUTPUT_DIR = r"data/processed"


class StockDataPipeline:
    """
    Pipeline para manipulação de dados históricos de ações.
    """
    def __init__(self, companies_path: str = COMPANIES_PATH, output_dir: str = OUTPUT_DIR):
        self.companies_path = companies_path
        self.output_dir = output_dir
        self.df: Optional[pd.DataFrame] = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Pipeline inicializado. Dados de empresas: {self.companies_path}, Output: {self.output_dir}")

    def load_listed_companies(self) -> Optional[pd.DataFrame]:
        """
        Load B3-listed companies from a CSV file with validation.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the list of companies, or None if loading fails.
        """
        try:
            df = pd.read_csv(self.companies_path)
        except FileNotFoundError:
            print(f"❌ File not found: {self.companies_path}")
            return None
        except Exception as e:
            print(f"❌ Error while reading CSV: {e}")
            return None

        # Validate 'Ticker' column
        if "Ticker" not in df.columns:
            possible = [c for c in df.columns if "tick" in c.lower() or "code" in c.lower()]
            if possible:
                print("⚠️ Missing 'Ticker' column — maybe you meant:", possible)
            else:
                print("⚠️ 'Ticker' column not found in CSV.")
            return None

        print(f"✅ Loaded {len(df)} tickers.")
        self.df = df
        return df

    def choose_ticker(self, interactive: bool = True) -> Optional[str]:
        """
        Select a ticker from the loaded companies DataFrame.

        Parameters
        ----------
        interactive : bool, optional
            If True, prompts user for input; otherwise, selects the first ticker.

        Returns
        -------
        Optional[str]
            Selected ticker with '.SA' suffix, or None if invalid.
        """
        if self.df is None:
            print("❌ No company data loaded. Call load_listed_companies first.")
            return None

        tickers = self.df["Ticker"].astype(str).str.upper().unique()

        if not interactive:
            print(f"Non-interactive mode: using {tickers[0]}")
            return tickers[0] + ".SA"

        print("Example tickers (first 20):")
        print(", ".join(tickers[:20]))
        ticker = input("Enter ticker (no .SA) or index (e.g. 0): ").strip().upper()

        if ticker.isdigit():
            idx = int(ticker)
            if 0 <= idx < len(tickers):
                return tickers[idx] + ".SA"
            else:
                print("Index out of range.")
                return None

        if ticker in tickers:
            return ticker + ".SA"

        print("Ticker not found.")
        return None

    def download_full_history(self, ticker: str, timeout: int = 20) -> Optional[pd.DataFrame]:
        """
        Download historical stock data for a given ticker.

        Parameters
        ----------
        ticker : str
            The stock ticker (e.g., 'PETR4.SA').
        timeout : int, optional
            Timeout for the download request in seconds.

        Returns
        -------
        Optional[pd.DataFrame]
            Historical data DataFrame, or None if download fails.
        """
        try:
            print(f"⬇️ Baixando {ticker} ...")
            data = yf.download(
                tickers=ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=True,
                threads=True,
                group_by="column",
                timeout=timeout
            )
            if data.empty:
                print("⚠️ Nenhum dado retornado.")
                return None

            # Normalize MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(0)
            
            # Ensure index is datetime
            data.index = pd.to_datetime(data.index)
            
            # Standardize column names
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            
            # Keep the datetime index (do not reset/drop Date column)
            if not data.empty and isinstance(data.index, pd.DatetimeIndex):
                print(f"✅ {len(data)} registros de {data.index.min().date()} a {data.index.max().date()}")
            else:
                print(f"✅ {len(data)} registros, mas o índice não é datetime ou está vazio.")
            
            return data
        except Exception as e:
            print(f"❌ Erro no download: {type(e).__name__} - {e}")
            return None

    def mark_and_remove_outliers(self, df: pd.DataFrame, multiplier: float = 1.5, verbose: bool = True) -> pd.DataFrame:
        """
        Mark and remove outliers from numerical columns in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        multiplier : float, optional
            Multiplier for IQR to determine outlier bounds.
        verbose : bool, optional
            If True, prints summary of outlier removal.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers removed and no outlier columns.
        """
        df_clean = df.copy()
        
        # Flatten MultiIndex if present
        if isinstance(df_clean.columns, pd.MultiIndex):
            df_clean.columns = ['_'.join(map(str, col)).strip() for col in df_clean.columns.values]

        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            if verbose:
                print("⚠️ Nenhuma coluna numérica encontrada")
            return df_clean

        outlier_cols = []
        for col in num_cols:
            col_data = df_clean[col]
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR

            # Create outlier column
            outlier_col = f"{col}_outlier"
            df_clean[outlier_col] = ((col_data < lower) | (col_data > upper)).astype(int)
            outlier_cols.append(outlier_col)

        # Remove rows with any outliers
        before = len(df_clean)
        mask = df_clean[outlier_cols].sum(axis=1) == 0
        df_clean = df_clean[mask]
        after = len(df_clean)

        # Drop outlier columns
        df_clean.drop(columns=outlier_cols, inplace=True)

        if verbose:
            removed = before - after
            print(f"Outlier removal completed:")
            print(f"  Columns checked: {len(num_cols)}")
            print(f"  Rows before: {before}")
            print(f"  Rows after:  {after}")
            print(f"  Removed:     {removed} ({removed/before*100:.2f}%)")

        return df_clean

    def export_data(self, data: pd.DataFrame) -> Optional[str]:
        """
        Export the DataFrame to a CSV file named 'df.csv'.

        Parameters
        ----------
        data : pd.DataFrame
            The data to export.

        Returns
        -------
        Optional[str]
            Filepath of the saved file, or None if export fails.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, "df.csv")

            # Flatten MultiIndex columns if any
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(0)

            data.to_csv(filepath, index=True)
            print(f"💾 Data successfully saved at: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Failed to export data: {e}")
            return None

    def run(self, ticker: Optional[str] = None, interactive: bool = True) -> Optional[pd.DataFrame]:
        """
        Run the full stock data pipeline for a given ticker.

        Parameters
        ----------
        ticker : Optional[str]
            The stock ticker (e.g., 'PETR4' or 'PETR4.SA'). If None, choose_ticker is called.
        interactive : bool, optional
            If True and ticker is None, prompts user to select a ticker.

        Returns
        -------
        Optional[pd.DataFrame]
            Processed DataFrame, or None if any step fails.
        """
        # Step 1: Load listed companies
        df_listed = self.load_listed_companies()
        if df_listed is None:
            return None

        # Step 2: Select ticker
        if ticker is None:
            ticker = self.choose_ticker(interactive=interactive)
        else:
            # Ensure ticker has .SA suffix and is valid
            ticker = ticker.upper()
            if not ticker.endswith(".SA"):
                ticker += ".SA"
            tickers = df_listed["Ticker"].astype(str).str.upper().unique()
            if ticker.replace(".SA", "") not in tickers:
                print(f"❌ Ticker {ticker} not found in listed companies.")
                return None

        if ticker is None:
            return None

        # Step 3: Download historical data
        df_hist = self.download_full_history(ticker)
        if df_hist is None:
            return None

        # Step 4: Remove outliers
        df_hist = self.mark_and_remove_outliers(df_hist)

        # Step 5: Export data
        self.export_data(df_hist)

        return df_hist


if __name__ == "__main__":
    pipeline = StockDataPipeline()
    # Example: Run pipeline with a specific ticker
    pipeline.run(ticker="PETR4", interactive=False)