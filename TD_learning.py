import tensorflow as tf
import numpy as np

# Parametri
input_dim = 4  # Dimensione dello stato
output_dim = 2  # Numero di azioni possibili
learning_rate = 0.001
discount_factor = 0.9  # Fattore di sconto
lambda_value = 0.5  # Parametro lambda per la traccia temporale (eligibility trace)

# Definisci il modello Q
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(output_dim, activation='linear')  # Lineare perché stiamo stimando i valori Q
])

# Definisci l'ottimizzatore
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Funzione di costo TD(λ)
def td_lambda_loss(y_true, y_pred, advantages, eligibility_trace):
    # Calcola la differenza temporale dell'errore
    temporal_difference = y_true - y_pred

    # Calcola l'errore pesato per la traccia temporale
    weighted_temporal_difference = temporal_difference * eligibility_trace

    # Calcola la loss TD(λ)
    loss = tf.reduce_mean(tf.square(weighted_temporal_difference) * advantages)

    return loss

# Esegui una passata in avanti (forward pass) per ottenere i valori Q predetti
state = tf.constant([[1.0, 2.0, 3.0, 4.0]])  # Esempio di stato
with tf.GradientTape(persistent=True) as tape:
    q_values = model(state)

# Calcola l'errore TD(λ) e il gradiente rispetto ai pesi
with tape:
    # Esempio di valori target (puoi sostituire questo con i tuoi valori target reali)
    target_q_values = tf.constant([[0.0, 1.0]])
    
    # Calcola l'errore TD(λ)
    temporal_difference = target_q_values - q_values
    
    # Calcola la traccia temporale (eligibility trace) utilizzando la lambda_value
    eligibility_trace = np.array([lambda_value ** i for i in range(len(temporal_difference))])
    
    # Calcola la loss TD(λ)
    loss = td_lambda_loss(target_q_values, q_values, temporal_difference, eligibility_trace)

# Calcola il gradiente della funzione di costo rispetto ai parametri del modello
gradients = tape.gradient(loss, model.trainable_variables)

# Applica l'aggiornamento dei pesi utilizzando l'ottimizzatore
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
