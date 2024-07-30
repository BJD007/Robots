def convert_vit_to_tensorrt(model_path, output_path):
    # Load your trained ViT model
    vit_model = tf.saved_model.load(model_path)
    
    # Create a TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create a network definition
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Parse the ONNX file
    parser = trt.OnnxParser(network, TRT_LOGGER)
    success = parser.parse_from_file(model_path)
    
    # Build the engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_engine(network, config)
    
    # Serialize and save the engine
    with open(output_path, "wb") as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {output_path}")

# Usage
convert_vit_to_tensorrt("path/to/vit_model.pb", "vit_tensorrt.engine")
