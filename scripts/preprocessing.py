import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TratamentoOHLCV:
    """
    Classe especializada para tratamento de outliers em dados OHLCV
    mantendo a integridade e relações entre as colunas
    """
    
    def __init__(self, df):
        """
        df deve conter: Open, High, Low, Close, Volume
        e idealmente um índice datetime
        """
        self.df = df.copy()
        self.df_original = df.copy()
        
    def validar_consistencia_ohlc(self):
        """
        Verifica se os dados OHLC mantêm suas relações lógicas
        High >= Open, Close, Low
        Low <= Open, Close, High
        """
        inconsistencias = pd.DataFrame()
        
        # High deve ser o maior
        mask_high = (self.df['High'] < self.df['Open']) | \
                    (self.df['High'] < self.df['Close']) | \
                    (self.df['High'] < self.df['Low'])
        
        # Low deve ser o menor
        mask_low = (self.df['Low'] > self.df['Open']) | \
                   (self.df['Low'] > self.df['Close']) | \
                   (self.df['Low'] > self.df['High'])
        
        inconsistencias['high_invalido'] = mask_high
        inconsistencias['low_invalido'] = mask_low
        
        total_problemas = mask_high.sum() + mask_low.sum()
        
        print(f"=== VALIDAÇÃO DE CONSISTÊNCIA OHLC ===")
        print(f"Registros com High inválido: {mask_high.sum()}")
        print(f"Registros com Low inválido: {mask_low.sum()}")
        print(f"Total de inconsistências: {total_problemas}")
        
        if total_problemas > 0:
            print("\n⚠️ ATENÇÃO: Dados inconsistentes detectados!")
            print("Essas linhas precisam de correção antes de prosseguir.")
        
        return inconsistencias
    
    def corrigir_inconsistencias_ohlc(self):
        """
        Corrige automaticamente inconsistências óbvias mantendo a lógica OHLC
        """
        df = self.df.copy()
        
        # Para cada linha, garante que High é o maior e Low é o menor
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        self.df = df
        print("✓ Inconsistências OHLC corrigidas")
        return self.df
    
    def detectar_outliers_price_change(self, threshold_percentual=10):
        """
        Detecta outliers baseado em mudança percentual extrema
        Mais apropriado que valores absolutos para preços
        """
        # Calcula retorno percentual
        retorno = self.df['Close'].pct_change() * 100
        
        # Detecta mudanças extremas
        outliers_mask = abs(retorno) > threshold_percentual
        
        print(f"\n=== DETECÇÃO POR MUDANÇA DE PREÇO ===")
        print(f"Threshold: {threshold_percentual}%")
        print(f"Outliers detectados: {outliers_mask.sum()} ({outliers_mask.sum()/len(self.df)*100:.2f}%)")
        
        if outliers_mask.sum() > 0:
            print(f"\nMaior alta: {retorno.max():.2f}%")
            print(f"Maior queda: {retorno.min():.2f}%")
        
        return outliers_mask
    
    def detectar_outliers_volume_zscore(self, threshold=3):
        """
        Detecta volumes anormais usando Z-Score
        Volume tende a ter distribuição log-normal
        """
        # Usa log do volume para normalizar
        log_volume = np.log1p(self.df['Volume'])
        z_scores = np.abs((log_volume - log_volume.mean()) / log_volume.std())
        
        outliers_mask = z_scores > threshold
        
        print(f"\n=== DETECÇÃO DE VOLUME ANORMAL ===")
        print(f"Threshold Z-Score: {threshold}")
        print(f"Outliers de volume: {outliers_mask.sum()} ({outliers_mask.sum()/len(self.df)*100:.2f}%)")
        
        return outliers_mask
    
    def detectar_gaps_anormais(self, threshold_percentual=5):
        """
        Detecta GAPs (diferença entre Close anterior e Open atual)
        GAPs podem ser legítimos, mas extremos indicam problemas
        """
        gap = ((self.df['Open'] - self.df['Close'].shift(1)) / 
               self.df['Close'].shift(1) * 100)
        
        outliers_mask = abs(gap) > threshold_percentual
        
        print(f"\n=== DETECÇÃO DE GAPS ANORMAIS ===")
        print(f"Threshold: {threshold_percentual}%")
        print(f"GAPs anormais: {outliers_mask.sum()}")
        
        return outliers_mask, gap
    
    def detectar_velas_impossíveis(self):
        """
        Detecta velas com características impossíveis ou muito suspeitas
        - Range (High-Low) muito maior que o normal
        - Volume zero em dias úteis
        """
        # Range da vela
        candle_range = ((self.df['High'] - self.df['Low']) / 
                        self.df['Close'] * 100)
        
        # Calcula IQR do range
        Q1 = candle_range.quantile(0.25)
        Q3 = candle_range.quantile(0.75)
        IQR = Q3 - Q1
        
        # Range anormal
        range_outliers = candle_range > (Q3 + 3 * IQR)
        
        # Volume zero (suspeito em dias úteis)
        volume_zero = self.df['Volume'] == 0
        
        print(f"\n=== DETECÇÃO DE VELAS IMPOSSÍVEIS ===")
        print(f"Velas com range anormal: {range_outliers.sum()}")
        print(f"Dias com volume zero: {volume_zero.sum()}")
        
        return range_outliers | volume_zero
    
    def tratar_interpolacao_temporal(self, outliers_mask):
        """
        MELHOR MÉTODO para dados financeiros!
        Usa interpolação temporal para manter continuidade da série
        """
        df = self.df.copy()
        
        # Para cada coluna OHLC
        for col in ['Open', 'High', 'Low', 'Close']:
            # Marca outliers como NaN
            df.loc[outliers_mask, col] = np.nan
            
            # Interpola linearmente
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # Volume: usa forward fill ou média móvel
        df.loc[outliers_mask, 'Volume'] = np.nan
        df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(5, min_periods=1).mean())
        
        # Garante consistência OHLC após interpolação
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        self.df = df
        print(f"\n✓ Interpolação temporal aplicada em {outliers_mask.sum()} registros")
        return self.df
    
    def tratar_winsorization_percentual(self, percentil_inferior=1, percentil_superior=99):
        """
        Winsorization adaptada: aplica apenas ao Close e ajusta OHLC proporcionalmente
        Mantém a estrutura da vela
        """
        df = self.df.copy()
        
        # Calcula limites apenas para Close
        lim_inf = df['Close'].quantile(percentil_inferior/100)
        lim_sup = df['Close'].quantile(percentil_superior/100)
        
        # Identifica outliers
        outliers_mask = (df['Close'] < lim_inf) | (df['Close'] > lim_sup)
        
        if outliers_mask.sum() > 0:
            # Calcula fator de ajuste proporcional
            for idx in df[outliers_mask].index:
                close_original = df.loc[idx, 'Close']
                close_ajustado = np.clip(close_original, lim_inf, lim_sup)
                
                # Fator de escala
                fator = close_ajustado / close_original if close_original != 0 else 1
                
                # Aplica proporcionalmente a OHLC
                df.loc[idx, 'Open'] *= fator
                df.loc[idx, 'High'] *= fator
                df.loc[idx, 'Low'] *= fator
                df.loc[idx, 'Close'] = close_ajustado
            
            self.df = df
            print(f"\n✓ Winsorization proporcional aplicada em {outliers_mask.sum()} registros")
        
        return self.df
    
    def tratar_marcacao_flag(self, outliers_mask):
        """
        Ao invés de alterar, apenas marca outliers com uma flag
        Útil para análise posterior ou filtros em modelos
        """
        self.df['is_outlier'] = outliers_mask
        self.df['outlier_type'] = ''
        
        print(f"\n✓ {outliers_mask.sum()} outliers marcados com flag 'is_outlier'")
        return self.df
    
    def pipeline_completo(self, 
                         corrigir_inconsistencias=True,
                         threshold_mudanca=10,
                         metodo='interpolacao'):
        """
        Pipeline completo de tratamento
        
        Parâmetros:
        - corrigir_inconsistencias: corrige High/Low inválidos
        - threshold_mudanca: % de mudança para considerar outlier
        - metodo: 'interpolacao', 'winsorization', ou 'flag'
        """
        print("="*70)
        print("PIPELINE DE TRATAMENTO DE OUTLIERS PARA DADOS OHLCV")
        print("="*70)
        
        # Passo 1: Validação
        inconsistencias = self.validar_consistencia_ohlc()
        
        if corrigir_inconsistencias and (inconsistencias.sum().sum() > 0):
            self.corrigir_inconsistencias_ohlc()
        
        # Passo 2: Detecção múltipla
        outliers_preco = self.detectar_outliers_price_change(threshold_mudanca)
        outliers_volume = self.detectar_outliers_volume_zscore()
        outliers_gaps, _ = self.detectar_gaps_anormais()
        outliers_velas = self.detectar_velas_impossíveis()
        
        # Combina todas as detecções
        outliers_final = outliers_preco | outliers_volume | outliers_gaps | outliers_velas
        
        print(f"\n{'='*70}")
        print(f"TOTAL DE OUTLIERS DETECTADOS: {outliers_final.sum()} "
              f"({outliers_final.sum()/len(self.df)*100:.2f}%)")
        print(f"{'='*70}")
        
        # Passo 3: Tratamento
        if metodo == 'interpolacao':
            self.tratar_interpolacao_temporal(outliers_final)
        elif metodo == 'winsorization':
            self.tratar_winsorization_percentual()
        elif metodo == 'flag':
            self.tratar_marcacao_flag(outliers_final)
        
        print("\n✅ Pipeline completo executado com sucesso!")
        return self.df


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

# Criando dados de exemplo
np.random.seed(42)

df = pd.read_csv(r"data\processed\df.csv", 
                 index_col=0, 
                 parse_dates=True,
                     dtype={
                        "Open": float,
                        "High": float,
                        "Low": float,
                        "Close": float,
                        "Volume": float,  # ou int, se quiser
                    }).reset_index()

# Usa o pipeline
tratador = TratamentoOHLCV(df)
df_tratado = tratador.pipeline_completo(
    corrigir_inconsistencias=True,
    threshold_mudanca=10,
    metodo='interpolacao'  # Recomendado para séries temporais financeiras
)

df_tratado.to_csv(r"data\processed\df.csv", index=True)

print("\n" + "="*70)
print("Dataset tratado pronto para uso!")
print("="*70)