import tensorflow as tf

class MetadataModel(tf.keras.Model):
    def train_step(self, data):
        # data = ({'input_field': x, 'metadata': meta}, y)
        x_dict, y = data
        img_input = x_dict['input_field']
        return super().train_step((img_input, y))

    def test_step(self, data):
        x_dict, y = data
        return super().test_step((x_dict['input_field'], y))

    def predict_step(self, data):
        x = data['input_field'] if isinstance(data, dict) else data
        return super().predict_step(x)
