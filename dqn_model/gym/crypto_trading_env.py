import gym
from gym import spaces
from gym.envs.registration import register


class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(CryptoTradingEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Пример: 0 - ничего не делать, 1 - покупать, 2 - продавать
        self.observation_space = spaces.Box(low=0, high=1, shape=(n,), dtype=np.float32)  # Предположим, n характеристик
        self.df = df
        self.current_step = 0

    def step(self, action):
        # Проверяем, достаточно ли данных осталось для выполнения анализа последствий действия
        done = self.current_step >= len(self.df) - 20
        reward = 0

        # Только если выполнение действия возможно, продолжаем
        if not done:
            # Получаем данные последних 10 свечей для текущего состояния
            state = self.df.iloc[self.current_step:self.current_step + 10].values

            # Вычисляем награду на основе действия и доступных данных
            reward = self.calculate_reward(action, self.df.iloc[self.current_step:self.current_step + 10])

            # Переходим на следующий шаг
            self.current_step += 1

            # Обновляем состояние завершения, если после текущего действия не осталось достаточно данных для следующего шага
            done = self.current_step >= len(self.df) - 20
        else:
            # Если мы достигли конца данных, состояние не определено
            state = None

        return state, reward, done, {}

    def calculate_reward(self, action, data):
        """Расчет награды на основе действия и данных последующих 10 свечей."""
        if action == 1:  # Купить
            future_price = self.df.iloc[self.current_step + 10]['close'] if self.current_step + 20 < len(self.df) else data.iloc[-1]['close']
            initial_price = data.iloc[-1]['close']
            reward = (future_price - initial_price) / initial_price  # Процент изменения цены
        return reward

    def reset(self):
        # Сброс состояния среды к начальному
        self.current_step = 0
        return self.df.iloc[self.current_step:self.current_step+10].values

    def render(self, mode='human', close=False):
        # Визуализация текущего состояния среды (опционально)
        pass

    def close(self):
        # Очистка при закрытии среды
        pass


register(
    id='CryptoTradingEnv-v0',
    entry_point='crypto_trading_env:CryptoTradingEnv',  # Обратите внимание на согласование имени файла и класса
)