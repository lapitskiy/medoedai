from utils.models_list import ModelLSTM_2Class

class SettingsRgr():
    CPU_COUNT = 2
    tfGPU = True
    goLSTM = True
    goKeras = True
    date_df = ['2024-03', '2024-04', '2024-05', ]
    coin = 'TONUSDT'
    numeric = ['open', 'high', 'low', 'close', 'bullish_volume', 'bearish_volume']
    checkpoint_file = 'temp/checkpoint/grid_search_checkpoint.txt'
    directory_save = None
    metric_thresholds = {
        'val_accuracy': 0.60,
        'val_precision': 0.60,
        'val_recall': 0.60,
        'val_f1_score': 0.10,
        'val_auc': 0.60
    }
    period = ["5m",]
    window_size = [3, 5, 8, 13, 21, 34]
    threshold = [0.005, 0.007, 0.01, 0.02]
    neiron = [50, 100, 150, 200]
    dropout = [0.10, 0.15, 0.20, 0.25, 0.30]
    model_count = ModelLSTM_2Class.model_count
    uuid = None

    def __init__(self):
        pass

class SettingsDqn():
    CPU_COUNT = 4
    goDQN = True
    ENV_NAME = "CryptoTradingEnv-v0"

    GAMMA = 0.95
    LEARNING_RATE = 0.001
    MEMORY_SIZE = 1000000
    BATCH_SIZE = 20
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

config = SettingsRgr()
lstmcfg = SettingsRgr()
dqncfg = SettingsDqn()
