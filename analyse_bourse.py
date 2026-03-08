"""
Analyse Bourse - IA pour la prédiction de cours boursiers
Combine analyse technique (ML) et analyse de sentiment (NLP)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class Config:
    """Configuration de l'application"""
    ticker: str = "MSFT"
    telegram_bot_token: str = "8789247732:AAH2FlYKcNneVrfqdwGIPsC-LBFeYm98ODk"
    telegram_chat_id: str = "6120802721"
    model_path: str = "model_rf.joblib"
    scaler_path: str = "scaler.joblib"
    lookback_period: str = "2y"
    interval: str = "1d"

config = Config()

# ==========================================
# 2. MODÈLE GLOBAL (Singleton pour FinBERT)
# ==========================================
class SentimentAnalyzer:
    """Analyseur de sentiment avec modèle chargé une seule fois"""
    
    _instance: Optional[Any] = None
    _nlp: Optional[Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._nlp is None:
            self._load_model()
    
    def _load_model(self):
        """Charge le modèle FinBERT une seule fois"""
        try:
            from transformers import BertTokenizer, BertForSequenceClassification, pipeline
            logger.info("Chargement du modèle FinBERT...")
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
            self._nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            logger.info("Modèle FinBERT chargé avec succès!")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self._nlp = None
    
    def analyze(self, news: List[Dict]) -> float:
        """Analyse le sentiment des nouvelles"""
        if not news or self._nlp is None:
            logger.warning("Pas de nouvelles ou modèle non chargé")
            return 0.0
        
        scores = []
        for n in news[:5]:  # Limite à 5 nouvelles
            try:
                title = n.get('title') or n.get('headline') or n.get('text', '')
                if title:
                    res = self._nlp(title)[0]
                    if res['label'] == 'Positive':
                        scores.append(1)
                    elif res['label'] == 'Negative':
                        scores.append(-1)
                    else:
                        scores.append(0)
            except Exception as e:
                logger.warning(f"Erreur analyse sentiment: {e}")
                continue
        
        return sum(scores) / len(scores) if scores else 0.0

# Instance globale
sentiment_analyzer = SentimentAnalyzer()

# ==========================================
# 3. COLLECTE ET INDICATEURS TECHNIQUES
# ==========================================
def get_data(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Télécharge les données et calcule les indicateurs techniques
    """
    try:
        logger.info(f"Téléchargement des données pour {ticker}...")
        
        # Méthode alternative avec start/end pour plus de fiabilité
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # ~2 ans
        
        # Essayer d'abord avec la méthode standard
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Si vide, essayer avec start/end
        if df.empty:
            logger.info("Méthode standard échouée, essai avec start/end...")
            df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'), 
                           progress=False)
        
        # Si toujours vide, essayer avec Ticker
        if df.empty:
            logger.info("Essai avec yf.Ticker...")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period="2y")
        
        if df.empty:
            logger.error(f"Aucune donnée trouvée pour {ticker}")
            return None
        
        # Indicateurs techniques
        df = _calculate_indicators(df)
        
        # Target: 1 si le cours monte demain, 0 sinon
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        logger.info(f"Données téléchargées: {len(df)} lignes")
        return df.dropna()
    
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        return None


def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule tous les indicateurs techniques"""
    
    # Extraction des colonnes Flatten pour éviter les erreurs MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0]  # Prendre la première colonne Close
        volume = df['Volume'].iloc[:, 0]
    else:
        close = df['Close']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    # RSI avec protection division par zéro
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    # Protection contre division par zéro
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Valeur neutre si pas de données
    
    # SMA (Simple Moving Averages)
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    df['SMA_200'] = close.rolling(200).mean()
    
    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # Calcul de BB_position avec protection division par zéro
    bb_range = bb_upper - bb_lower
    df['BB_position'] = np.where(bb_range != 0, (close - bb_lower) / bb_range, 0.5)
    
    # Volatilité
    df['Volatility'] = close.rolling(20).std()
    
    # Returns
    df['Returns'] = close.pct_change()
    df['Returns_5d'] = close.pct_change(5)
    
    # Volume moyenne
    if 'Volume' in df.columns:
        df['Volume_MA'] = volume.rolling(20).mean()
    
    return df

# ==========================================
# 4. MACHINE LEARNING
# ==========================================
def train_and_predict(df: pd.DataFrame) -> tuple[int, float]:
    """
    Entraîne le modèle et fait une prédiction
    Returns: (prédiction, probabilité)
    """
    features = [
        'Close', 'SMA_20', 'RSI', 'SMA_50', 'MACD', 
        'MACD_signal', 'MACD_hist', 'BB_position', 
        'Volatility', 'Returns', 'Returns_5d'
    ]
    
    # Vérification que toutes les features existent
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df['Target']
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sauvegarde du scaler
    try:
        joblib.dump(scaler, config.scaler_path)
        logger.info("Scaler sauvegardé")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le scaler: {e}")
    
    # Entraînement (sauf dernière ligne pour prédiction)
    X_train = X_scaled[:-1]
    y_train = y[:-1]
    
    # Si pas assez de données, utiliser tout
    if len(X_train) < 50:
        logger.warning("Peu de données, entraînement sur tout le dataset")
        X_train = X_scaled
        y_train = y
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Sauvegarde du modèle
    try:
        joblib.dump(model, config.model_path)
        logger.info("Modèle sauvegardé")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le modèle: {e}")
    
    # Prédiction sur la dernière ligne
    X_pred = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(X_pred)[0]
    probability = model.predict_proba(X_pred)[0]
    
    return int(prediction), float(probability[1])  # Probabilité de classe 1 (hausse)


def load_model() -> Optional[Any]:
    """Charge un modèle existant"""
    if os.path.exists(config.model_path):
        try:
            return joblib.load(config.model_path)
        except Exception as e:
            logger.warning(f"Impossible de charger le modèle: {e}")
    return None

# ==========================================
# 5. TELEGRAM
# ==========================================
def send_telegram_alert(
    ticker: str, 
    pred: int, 
    sent: float, 
    rsi: float,
    probability: float = 0.5
) -> bool:
    """Envoie une alerte Telegram"""
    
    # Déterminer le conseil
    if pred == 1 and sent > 0 and rsi < 70:
        conseil = "ACHETER 🚀"
    elif pred == 0 or rsi > 70:
        conseil = "VENDRE/ATTENDRE ✋"
    else:
        conseil = "SURVEILLER 🧐"
    
    # Formatage du message
    confidence = probability * 100
    message = f"""
📊 <b>Rapport IA - {ticker}</b>
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}

🤖 <b>Prédiction IA:</b> {'Hausse ⬆️' if pred == 1 else 'Baisse ⬇️'}
📊 <b>Confiance:</b> {confidence:.1f}%

💭 <b>Score Sentiment:</b> {sent:.2f}
📈 <b>RSI actuel:</b> {rsi:.2f}

💡 <b>Conseil:</b> {conseil}

⚠️ <i>Disclaimer: Ce n'est pas un conseil financier!</i>
"""
    
    # Vérifier si les identifiants sont configurés
    if (config.telegram_bot_token == "votre_token_bot_telegram" or 
        config.telegram_chat_id == "votre_chat_id"):
        logger.warning("Config Telegram non configurée - affichage du message:")
        print(message)
        return False
    
    try:
        import requests
        url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": config.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("Message Telegram envoyé avec succès!")
            return True
        else:
            logger.error(f"Erreur Telegram: {response.text}")
            return False
            
    except ImportError:
        logger.warning("Requests non installé - affichage du message:")
        print(message)
        return False
    except Exception as e:
        logger.error(f"Erreur Telegram: {e}")
        return False

# ==========================================
# 6. CLASSE PRINCIPALE
# ==========================================
class BourseAnalyzer:
    """Analyseur boursier complet"""
    
    def __init__(self, ticker: str = "MSFT"):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data: Optional[pd.DataFrame] = None
        self.prediction: Optional[int] = None
        self.sentiment: Optional[float] = None
        self.rsi: Optional[float] = None
        self.probability: float = 0.5
    
    def run(self) -> bool:
        """Exécute l'analyse complète"""
        logger.info(f"=== Analyse pour {self.ticker} ===")
        
        # 1. Données
        self.data = get_data(self.ticker)
        if self.data is None:
            return False
        
        # 2. Sentiment
        try:
            news = self.stock.news if hasattr(self.stock, 'news') else []
            self.sentiment = sentiment_analyzer.analyze(news)
        except Exception as e:
            logger.warning(f"Erreur sentiment: {e}")
            self.sentiment = 0.0
        
        # 3. Prédiction ML
        try:
            self.prediction, self.probability = train_and_predict(self.data)
            self.rsi = self.data['RSI'].iloc[-1]
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            return False
        
        # 4. Notification
        send_telegram_alert(
            self.ticker, 
            self.prediction, 
            self.sentiment, 
            self.rsi,
            self.probability
        )
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'analyse"""
        return {
            'ticker': self.ticker,
            'prediction': 'Hausse' if self.prediction == 1 else 'Baisse',
            'confidence': f"{self.probability * 100:.1f}%",
            'sentiment': self.sentiment,
            'rsi': self.rsi,
            'price': self.data['Close'].iloc[-1] if self.data is not None else None
        }

# ==========================================
# 7. FONCTION PRINCIPALE
# ==========================================
def main(ticker: str = "MSFT", send_alert: bool = True):
    """Point d'entrée principal"""
    analyzer = BourseAnalyzer(ticker)
    success = analyzer.run()
    
    if success:
        summary = analyzer.get_summary()
        logger.info(f"Résumé: {summary}")
        print(f"\n✅ Analyse terminée pour {ticker}")
        print(f"   Prédiction: {summary['prediction']}")
        print(f"   Confiance: {summary['confidence']}")
    else:
        print(f"❌ Échec de l'analyse pour {ticker}")
    
    return success


# ==========================================
# 8. REQUIREMENTS
# ==========================================
def get_requirements() -> List[Dict[str, str]]:
    """
    Retourne la liste des dépendances nécessaires
    """
    return [
        {"name": "yfinance", "version": ">=0.2.0", "description": "Données boursières"},
        {"name": "pandas", "version": ">=1.5.0", "description": "Manipulation de données"},
        {"name": "numpy", "version": ">=1.21.0", "description": "Calculs numériques"},
        {"name": "scikit-learn", "version": ">=1.0.0", "description": "Machine Learning"},
        {"name": "joblib", "version": ">=1.2.0", "description": "Sauvegarde modèle"},
        {"name": "requests", "version": ">=2.28.0", "description": "API Telegram"},
        {"name": "transformers", "version": ">=4.30.0", "description": "FinBERT (NLP)"},
        {"name": "torch", "version": ">=2.0.0", "description": "Backend FinBERT"},
    ]


def requirements():
    """Affiche les dépendances requises"""
    print("\n" + "="*50)
    print("📦 DÉPENDANCES REQUISES")
    print("="*50)
    
    deps = get_requirements()
    for dep in deps:
        print(f"  • {dep['name']} {dep['version']}")
        print(f"    └─ {dep['description']}")
    
    print("\n💾 Installation:")
    print("  pip install -r requirements.txt")
    print("="*50 + "\n")


def generate_requirements_file(filename: str = "requirements.txt"):
    """Génère un fichier requirements.txt"""
    deps = get_requirements()
    
    content = "# Requirements - Analyse Bourse IA\n"
    content += "# Generated automatically\n\n"
    
    for dep in deps:
        content += f"{dep['name']}{dep['version']}\n"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fichier {filename} généré avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")


# ==========================================
# EXÉCUTION
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse Bourse IA')
    parser.add_argument('--ticker', type=str, default='MSFT', help='Symbole boursier')
    parser.add_argument('--no-alert', action='store_true', help='Désactiver les alertes Telegram')
    parser.add_argument('--requirements', action='store_true', help='Afficher les dépendances requises')
    parser.add_argument('--generate-requirements', action='store_true', help='Générer un fichier requirements.txt')
    
    args = parser.parse_args()
    
    if args.requirements:
        requirements()
    elif args.generate_requirements:
        generate_requirements_file()
    else:
        main(ticker=args.ticker.upper(), send_alert=not args.no_alert)

