

def trade_once(state: np.ndarray, observation_space_dim: int): 
    """
    Принимает торговое решение на основе текущего состояния.

    Args:
        state (np.ndarray): Текущий вектор состояния.
        observation_space_dim (int): Размерность пространства наблюдений (длина вектора состояния).
        model_path (str): Путь к файлу сохраненной модели.
    Returns:
        str: Рекомендованное торговое действие ("BUY", "SELL", "HOLD").
    """
    
    if not os.path.exists(cfg.model_path):
        return "Модель не найдена, сначала обучите её!"

    # Мы знаем, что action_space_dim = 3 (HOLD, BUY, SELL) из вашего Gym окружения.
    # Или можно получить это из окружения, если оно было зарегистрировано:
    # action_space_dim = gym.make(dqncfg.ENV_NAME).action_space.n 
    action_space_dim = 3 # Поскольку действия фиксированы, можно захардкодить или передать из dqncfg

    model = DQNN(observation_space_dim, action_space_dim).to(DEVICE)
    model.load_state_dict(torch.load(cfg.model_path, map_location=DEVICE))
    model.eval() # Переводим модель в режим оценки для предсказания

    # Преобразование состояния в тензор и отправка на устройство
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): # Отключаем вычисление градиентов для инференса
        q_values = model(state_tensor)
    
    action = torch.argmax(q_values[0]).item() # Получаем индекс действия

    actions_map = {0: "HOLD", 1: "BUY", 2: "SELL"} 
    executed_action = actions_map[action]

    print(f"Торговое действие: {executed_action} с Q-values {q_values.cpu().numpy()}")
    return executed_action