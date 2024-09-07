import open3d as o3d
import numpy as np
import tensorflow as tf

# Load the trained model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Load and preprocess the point cloud data from a .pcd file
def preprocess_point_cloud(pcd_filename, num_points=223436):  # Adjust num_points if needed
    # Read the .pcd file using Open3D
    pcd = o3d.io.read_point_cloud(pcd_filename)
    
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    # Check if there are colors available
    if len(np.asarray(pcd.colors)) > 0:
        colors = np.asarray(pcd.colors)
        # Concatenate XYZ and RGB (or just use XYZ if no colors)
        data = np.concatenate((points, colors), axis=1)
    else:
        data = points
    
    # Normalize or adjust the data if necessary
    # For instance, you might want to pad or crop to match num_points
    if data.shape[0] < num_points:
        # Pad the point cloud if it has fewer points
        data = np.pad(data, ((0, num_points - data.shape[0]), (0, 0)), mode='constant')
    elif data.shape[0] > num_points:
        # Crop the point cloud if it has more points
        data = data[:num_points]
    
    # Reshape to match the model input shape
    data = data.reshape((1, num_points, data.shape[1]))  # Add batch dimension

    return data

# Make predictions using the model
def infer_point_cloud(model, point_cloud_data):
    predictions = model.predict(point_cloud_data)
    return predictions

# Post-process and interpret the results
def postprocess_predictions(predictions):
    # Example: Convert predictions to bounding boxes or other formats
    return predictions

# Example usage
if __name__ == "__main__":
    model_path = 'pointnet_model.h5'
    pcd_filename = 'pointclouds/2.pcd'
    
    # Load the model
    model = load_model(model_path)
    
    # Print the model input shape
    input_shape = model.input_shape[1:]
    print(f"Model input shape: {input_shape}")
    
    # Load and preprocess point cloud data
    point_cloud_data = preprocess_point_cloud(pcd_filename)
    
    # Check the shape of the point cloud data
    print(f"Point Cloud Data shape: {point_cloud_data.shape}")
    
    # Ensure the data shape matches the model's input shape
    if point_cloud_data.shape[1:] != input_shape:
        print("Shape mismatch: Adjusting point cloud data preprocessing.")
        # This is a safety check; adjust num_points or preprocessing as needed
        # For now, just print the shapes for debugging
        print(f"Expected input shape: {input_shape}")
        print(f"Actual point cloud data shape: {point_cloud_data.shape[1:]}")
    
    assert point_cloud_data.shape[1:] == input_shape, "Point cloud data shape does not match the model input shape."
    
    # Make predictions
    predictions = infer_point_cloud(model, point_cloud_data)
    
    # Post-process predictions
    processed_predictions = postprocess_predictions(predictions)
    
    # Print or visualize the results
    print(f"Predictions: {processed_predictions}")
