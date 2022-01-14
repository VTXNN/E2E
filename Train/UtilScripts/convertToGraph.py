import tensorflow as tf

weightModel = tf.keras.models.load_model('weightModel_unquantised')
patternModel = tf.keras.models.load_model('patternModel_unquantised')
associationModel = tf.keras.models.load_model("asociationModel_unquantised")

import cmsml
cmsml.tensorflow.save_graph("weightModelgraph.pb", weightModel, variables_to_constants=True)
cmsml.tensorflow.save_graph("patternModelgraph.pb", patternModel, variables_to_constants=True)
cmsml.tensorflow.save_graph("asociationModelgraph.pb", associationModel, variables_to_constants=True)
