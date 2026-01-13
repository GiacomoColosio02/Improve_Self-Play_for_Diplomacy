import tensorflow as tf
import numpy as np


class DilpILKNet:
    def __init__(self):
        self.encoder = DilpILKEncoder()
        self.policy_head = DilpILKPolicyHead()
        self.value_head = DilpILKValueHead()

    def call(self, board_state, training=False):
        embeddings = self.encoder(board_state)
        policy = self.policy_head(embeddings, training=training)
        value, weights = self.value_head(embeddings)
        return policy, value, weights


class DilpILKEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim=224, num_heads=8, ff_dim=224, num_layers=10, num_positions=512, dropout=0.0):
        super().__init__()
        self.player_state_layer = tf.keras.layers.Dense(embed_dim, use_bias=True, activation=None)
        self.location_state_layer = tf.keras.layers.Dense(embed_dim, use_bias=True, activation=None)
        self.global_state_layer = tf.keras.layers.Dense(embed_dim, use_bias=True, activation=None)

        key_dim = embed_dim // num_heads
        self.attn_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
            for _ in range(num_layers)
        ]
        self.proj_layer = tf.keras.layers.Dense(embed_dim)
        self.positional_bias = self.add_weight(
            shape=(num_positions, embed_dim),
            name="positional_bias",
            initializer="zeros",
            trainable=True,
            dtype=tf.float32
        )

        self.attn_norm_layers = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.ffn_layers = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation='gelu'),
                tf.keras.layers.Dense(embed_dim)
            ]) for _ in range(num_layers)
        ]
        self.ffn_norm_layers = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

    def call(self, board_state):
        player_state = self.player_state_layer(board_state["player"])
        location_state = self.location_state_layer(board_state["location"])
        global_state = self.global_state_layer(board_state["global"])

        x = tf.concat([location_state, player_state, global_state], axis=1)
        x = self.proj_layer(x)

        seq_len = tf.shape(x)[1]
        pos_bias = self.positional_bias[:seq_len, :]
        x = x + tf.expand_dims(pos_bias, axis=0)

        for i in range(len(self.attn_layers)):
            y = self.attn_norm_layers[i](x)
            attn_output = self.attn_layers[i](y, y)
            x = x + attn_output

            y = self.ffn_norm_layers[i](x)
            ffn_output = self.ffn_layers[i](y)
            x = x + ffn_output

        embeddings = x
        return embeddings


class DilpILKPolicyHead(tf.keras.layers.Layer):
    def __init__(self, num_actions_per_unit=10):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(
            200, return_sequences=True, return_state=True
        )
        self.lstm2 = tf.keras.layers.LSTM(
            200, return_sequences=True, return_state=True
        )
        self.output_layer = tf.keras.layers.Dense(num_actions_per_unit)

    def call(self, embeddings, training=False):
        """
        embeddings: [batch, num_units, embed_dim]
        Returns:
            policy_logits: [batch, num_units, num_actions_per_unit]
        """
        seq_out1, h1, c1 = self.lstm1(embeddings, training=training)
        seq_out2, h2, c2 = self.lstm2(seq_out1, training=training)
        logits = self.output_layer(seq_out2)
        return logits

        # policy_probs = tf.nn.softmax(logits, axis=-1)
        # return policy_probs


class DilpILKValueHead(tf.keras.layers.Layer):
    def __init__(self, embed_dim=224, num_players=7):
        super().__init__()
        # score_layer: maps each position embedding -> scalar logit
        self.score_layer = tf.keras.layers.Dense(1, name="value_pos_score")
        # MLP after pooling: 224 -> 224 -> num_players
        self.fc1 = tf.keras.layers.Dense(embed_dim, name="value_fc1")
        self.act = tf.keras.layers.Activation('gelu')  # paper uses GeLU
        self.fc2 = tf.keras.layers.Dense(num_players, name="value_fc2")

    def call(self, embeddings):
        """
        embeddings: [batch, seq_len, embed_dim]
        Returns:
            player_logits: [batch, num_players]
            weights: [batch, seq_len]  (positional attention weights)
        """
        # per-pos logits
        pos_logits = self.score_layer(embeddings)
        pos_logits = tf.squeeze(pos_logits, axis=-1)

        # softmax over positions to get weights
        weights = tf.nn.softmax(pos_logits, axis=1)

        # weighted sum to get pooled embedding
        weights_exp = tf.expand_dims(weights, axis=-1)
        pooled = tf.reduce_sum(weights_exp * embeddings, axis=1)

        # MLP -> per-player logits
        x = self.fc1(pooled)
        x = self.act(x)
        player_logits = self.fc2(x)

        return player_logits, weights


class BoardStateEncoder:
    def _encode_location_state(game):
        map_ = game.map
        locs = map_.locs
        players = list(game.powers.keys())
        player_index = {p: i for i, p in enumerate(players)}
        loc_features = np.zeros((1, len(locs), 38), dtype=np.float32)

        units_by_power = game.get_units(None)
        loc_to_unit = {
            u.split()[1].upper(): {
                "type": u.split()[0].strip('*'),
                "owner": power,
                "dislodged": u.startswith('*')
            }
            for power, units in units_by_power.items()
            for u in units
        }

        centers_by_power = game.get_centers(None)
        sc_owners = {
            loc.upper(): power
            for power, centers in centers_by_power.items()
            for loc in centers
        }

        dislodged_info = {}
        for unit_str, attacker in game.dislodged.items():
            parts = unit_str.strip('*').split()
            if len(parts) != 2:
                continue
            unit_type, loc = parts
            loc = loc.upper()
            owner = loc_to_unit.get(loc, {}).get("owner", None)
            dislodged_info[loc] = {
                "type": unit_type,
                "owner": owner,
                "attacker": attacker
            }

        for i, loc in enumerate(locs):
            loc_upper = loc.upper()
            unit_data = loc_to_unit.get(loc_upper)
            if unit_data and not unit_data.get("dislodged"):
                loc_features[0, i, 0] = 1.0 if unit_data["type"] == "A" else 0.0
                loc_features[0, i, 1] = 1.0 if unit_data["type"] == "F" else 0.0
                owner_idx = player_index.get(unit_data["owner"], -1)
                if owner_idx >= 0:
                    loc_features[0, i, 2 + owner_idx] = 1.0

            du = dislodged_info.get(loc_upper)
            if du:
                loc_features[0, i, 11] = 1.0 if du["type"] == "A" else 0.0
                loc_features[0, i, 12] = 1.0 if du["type"] == "F" else 0.0
                owner_idx = player_index.get(du["owner"], -1)
                if owner_idx >= 0:
                    loc_features[0, i, 13 + owner_idx] = 1.0

            sc_owner = sc_owners.get(loc_upper)
            if sc_owner:
                owner_idx = player_index.get(sc_owner, -1)
                if owner_idx >= 0:
                    loc_features[0, i, 23 + owner_idx] = 1.0

            loc_type = map_.loc_type.get(loc_upper)
            if loc_type == "land":
                loc_features[0, i, 20] = 1.0
            elif loc_type == "coast":
                loc_features[0, i, 21] = 1.0
            elif loc_type == "water":
                loc_features[0, i, 22] = 1.0

        return tf.convert_to_tensor(loc_features)

    def _encode_player_state(game):
        players = list(game.powers.keys())
        player_features = np.zeros((1, len(players), 1), dtype=np.float32)
        centers_by_power = game.get_centers(None)
        for i, p in enumerate(players):
            player_features[0, i, 0] = float(len(centers_by_power.get(p, [])))
        return tf.convert_to_tensor(player_features)

    def _encode_global_state(game):
        phase_parts = game.phase.split()
        season_map = {"SPRING": [1, 0, 0], "FALL": [0, 1, 0], "WINTER": [0, 0, 1]}
        season = phase_parts[0].upper() if len(phase_parts) > 0 else "SPRING"
        season_onehot = season_map.get(season, [0, 0, 0])
        year = (int(phase_parts[1]) - 1901) / 10.0 if len(phase_parts) > 1 else 0.0
        global_features = np.array([[season_onehot + [year, 1.0, 1.0, 0.0]]], dtype=np.float32)
        return tf.convert_to_tensor(global_features)

    def encode_board_state(game, location_state=None, global_state=None):
        location_state = BoardStateEncoder._encode_location_state(game)
        global_state = BoardStateEncoder._encode_global_state(game)
        player_state = BoardStateEncoder._encode_player_state(game)
        return {
            "player": player_state,
            "location": location_state,
            "global": global_state
        }