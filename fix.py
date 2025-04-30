import tensorflow as tf
import os
import shutil

# Custom layer definitions
@tf.keras.utils.register_keras_serializable()
class CompatibleLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove problematic argument
        super().__init__(*args, **kwargs)

def fix_model(model_path):
    try:
        # Create fixed models directory if it doesn't exist
        os.makedirs('fixed_models', exist_ok=True)
        
        # Custom objects for loading
        custom_objects = {
            'LSTM': CompatibleLSTM,
            'Bidirectional': tf.keras.layers.Bidirectional
        }
        
        # Load model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Save in both .keras and .h5 formats
        base_name = os.path.basename(model_path).replace('.h5', '')
        model.save(f'fixed_models/{base_name}.keras')
        model.save(f'fixed_models/{base_name}.h5')
        
        print(f"Successfully fixed {model_path}")
        return True
        
    except Exception as e:
        print(f"Failed to fix {model_path}: {str(e)}")
        return False

if __name__ == "__main__":
    # Process all .h5 files in models directory
    for model_file in os.listdir('models'):
        if model_file.endswith('.h5'):
            fix_model(f'models/{model_file}')