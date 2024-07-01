from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, Input, Bidirectional, BatchNormalization, \
    Conv1D, Attention, GRU, InputLayer, MultiHeadAttention
from tensorflow.keras.optimizers import Adam

class ModelLSTM_2Class:
    model_count = 10

    def __init__(self, model_number, current_window, num_features, current_neiron, current_dropout):
        self.model_number = model_number
        self.current_window = current_window
        self.num_features = num_features
        self.current_neiron = current_neiron
        self.current_dropout = current_dropout
        self.model = None

    def build_model(self):
        model = Sequential()
        if self.model_number == 1:
            model.add(Input(shape=(self.current_window, self.num_features)))
            model.add(LSTM(self.current_neiron, return_sequences=True))
            model.add(Dropout(self.current_dropout))
            model.add(LSTM(self.current_neiron, return_sequences=True))
            model.add(Dropout(self.current_dropout))  # Добавление слоя Dropout
            model.add(LSTM(self.current_neiron))
            model.add(Dropout(self.current_dropout))  # Добавление слоя Dropout
            model.add(Dense(1, activation='sigmoid'))
        if self.model_number == 2:
            model.add(Input(shape=(self.current_window, self.num_features)))
            model.add(LSTM(self.current_neiron, activation='relu', recurrent_activation='sigmoid', return_sequences=True))
            model.add(Dropout(self.current_dropout))
            model.add(LSTM(self.current_neiron, activation='relu', recurrent_activation='sigmoid'))
            model.add(Dropout(self.current_dropout))
            model.add(Dense(1, activation='sigmoid'))
        if self.model_number == 3:
            model.add(Input(shape=(self.current_window, self.num_features)))
            model.add(LSTM(self.current_neiron, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
            model.add(Dropout(self.current_dropout))
            model.add(LSTM(self.current_neiron, activation='tanh', recurrent_activation='sigmoid'))
            model.add(Dropout(self.current_dropout))
            model.add(Dense(1, activation='sigmoid'))
        if self.model_number == 4:
            model.add(Input(shape=(self.current_window, self.num_features)))
            model.add(LSTM(self.current_neiron, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
            model.add(Dropout(self.current_dropout))
            model.add(LSTM(self.current_neiron))
            model.add(LeakyReLU(alpha=0.01))
            model.add(Dense(1, activation='sigmoid'))
        if self.model_number == 5:
            model.add(Bidirectional(LSTM(self.current_neiron, return_sequences=True)))
            model.add(Dropout(self.current_dropout))
            model.add(Bidirectional(LSTM(self.current_neiron)))
            model.add(Dropout(self.current_dropout))
            model.add(Dense(1, activation='sigmoid'))
        if self.model_number == 6:
            model.add(LSTM(self.current_neiron, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(self.current_dropout))
            model.add(LSTM(self.current_neiron))
            model.add(BatchNormalization())
            model.add(Dropout(self.current_dropout))
            model.add(Dense(1, activation='sigmoid'))
        if self.model_number == 7:
            model.add(InputLayer(shape=(self.current_window, self.num_features)))
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
            model.add(Dropout(self.current_dropout))
            model.add(LSTM(self.current_neiron))
            model.add(Dropout(self.current_dropout))
            model.add(Dense(1, activation='sigmoid'))
        if self.model_number == 8:
            input_layer = Input(shape=(self.current_window, self.num_features))
            lstm_layer = LSTM(self.current_neiron, return_sequences=True)(input_layer)
            attention_output = MultiHeadAttention(num_heads=2, key_dim=self.current_neiron)(query=lstm_layer, key=lstm_layer, value=lstm_layer)
            attention_output = Dropout(self.current_dropout)(attention_output)
            lstm_layer_2 = LSTM(self.current_neiron)(attention_output)
            output_layer = Dense(1, activation='sigmoid')(lstm_layer_2)
            model = Model(inputs=input_layer, outputs=output_layer)
        if self.model_number == 9:
            input_layer = Input(shape=(self.current_window, self.num_features))
            lstm_layer = LSTM(self.current_neiron, return_sequences=True)(input_layer)
            attention_output = Attention()([lstm_layer, lstm_layer, lstm_layer])
            dropout_out = Dropout(self.current_dropout)(attention_output)
            lstm_out2 = LSTM(self.current_neiron)(dropout_out)
            output_layer = Dense(1, activation='sigmoid')(lstm_out2)
            model = Model(inputs=input_layer, outputs=output_layer)
        if self.model_number == 10:
            model.add(GRU(self.current_neiron, return_sequences=True))
            model.add(Dropout(self.current_dropout))
            model.add(GRU(self.current_neiron))
            model.add(Dropout(self.current_dropout))
            model.add(Dense(1, activation='sigmoid'))
        print(f'model num: {self.model_number}')
        self.model = model


### Grid model

def create_model(current_dropout=None, current_neiron=None, current_window=None, num_features=None, model_number=None, type=None):
    model = Sequential()
    if model_number == 1:
        model.add(Input(shape=(current_window, num_features)))
        model.add(LSTM(current_neiron, return_sequences=True))
        model.add(Dropout(current_dropout))
        model.add(LSTM(current_neiron, return_sequences=True))
        model.add(Dropout(current_dropout))  # Добавление слоя Dropout
        model.add(LSTM(current_neiron))
        model.add(Dropout(current_dropout))  # Добавление слоя Dropout
        model.add(Dense(1, activation='sigmoid'))
    if model_number == 2:
        model.add(Input(shape=(current_window, num_features)))
        model.add(LSTM(current_neiron, activation='relu', recurrent_activation='sigmoid', return_sequences=True))
        model.add(Dropout(current_dropout))
        model.add(LSTM(current_neiron, activation='relu', recurrent_activation='sigmoid'))
        model.add(Dropout(current_dropout))
        model.add(Dense(1, activation='sigmoid'))
    if model_number == 3:
        model.add(Input(shape=(current_window, num_features)))
        model.add(LSTM(current_neiron, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
        model.add(Dropout(current_dropout))
        model.add(LSTM(current_neiron, activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(current_dropout))
        model.add(Dense(1, activation='sigmoid'))
    if model_number == 4:
        model.add(Input(shape=(current_window, num_features)))
        model.add(LSTM(current_neiron, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
        model.add(Dropout(current_dropout))
        model.add(LSTM(current_neiron))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1, activation='sigmoid'))
    if model_number == 5:
        model.add(InputLayer(input_shape=(current_window, num_features)))  # Добавьте входной слой
        model.add(Bidirectional(LSTM(current_neiron, return_sequences=True)))
        model.add(Dropout(current_dropout))
        model.add(Bidirectional(LSTM(current_neiron)))
        model.add(Dropout(current_dropout))
        model.add(Dense(1, activation='sigmoid'))
    if model_number == 6:
        model.add(InputLayer(input_shape=(current_window, num_features)))  # Добавьте входной слой
        model.add(LSTM(current_neiron, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(current_dropout))
        model.add(LSTM(current_neiron))
        model.add(BatchNormalization())
        model.add(Dropout(current_dropout))
        model.add(Dense(1, activation='sigmoid'))
    if model_number == 7:
        model.add(InputLayer(shape=(current_window, num_features)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(Dropout(current_dropout))
        model.add(LSTM(current_neiron))
        model.add(Dropout(current_dropout))
        model.add(Dense(1, activation='sigmoid'))
    if model_number == 8:
        input_layer = Input(shape=(current_window, num_features))
        lstm_layer = LSTM(current_neiron, return_sequences=True)(input_layer)
        attention_output = MultiHeadAttention(num_heads=2, key_dim=current_neiron)(query=lstm_layer, key=lstm_layer, value=lstm_layer)
        attention_output = Dropout(current_dropout)(attention_output)
        lstm_layer_2 = LSTM(current_neiron)(attention_output)
        output_layer = Dense(1, activation='sigmoid')(lstm_layer_2)
        model = Model(inputs=input_layer, outputs=output_layer)
    if model_number == 9:
        input_layer = Input(shape=(current_window, num_features))
        lstm_layer = LSTM(current_neiron, return_sequences=True)(input_layer)
        attention_output = Attention()([lstm_layer, lstm_layer, lstm_layer])
        dropout_out = Dropout(current_dropout)(attention_output)
        lstm_out2 = LSTM(current_neiron)(dropout_out)
        output_layer = Dense(1, activation='sigmoid')(lstm_out2)
        model = Model(inputs=input_layer, outputs=output_layer)
    if model_number == 10:
        model.add(InputLayer(shape=(current_window, num_features)))
        model.add(GRU(current_neiron, return_sequences=True))
        model.add(Dropout(current_dropout))
        model.add(GRU(current_neiron))
        model.add(Dropout(current_dropout))
        model.add(Dense(1, activation='sigmoid'))
    # model = Sequential()
    # model.add(Input(shape=(current_window, num_features)))
    # model.add(LSTM(lstm_neurons))
    # model.add(Dense(1))
    # model.compile(optimizer=optimizer, loss='mse')
    if type == 'lstm':
        return model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model
'''
def create_greed_model(neurons=None, dropout_rate=None, model_number=None, current_window=None, num_features=None):
    model = Sequential()
    print(f'model_number {model_number}')
    if model_number == 1:
        model.add(Input(shape=(current_window, num_features)))
        model.add(LSTM(neurons, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(neurons, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(neurons))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model
'''