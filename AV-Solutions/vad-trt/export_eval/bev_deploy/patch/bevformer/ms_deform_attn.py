import torch
# from torch.onnx import register_custom_op_symbolic # Keep commented if using custom registration
import torch.onnx.symbolic_helper as sym_help # Useful for ONNX constants if needed

from ..custom_op import register

@register
def custom_ms_deform_attn_op_handler(g, *args, **kwargs): # args are op_input_0, op_input_1, ...
    node_name = kwargs.get("name")
    if not node_name:
        # Handle cases where name might not be passed, though your setup implies it is.
        # Or simply rely on it being there.
        return # Or raise error

    if node_name == "MultiScaleDeformableAttnFunction_fp32" or \
       node_name == "MultiScaleDeformableAttnFunction":

        # Assuming standard inputs for MSDeformAttn:
        # args[0]: value (Tensor)
        # args[1]: value_spatial_shapes (Tensor)
        # args[2]: value_level_start_index (Tensor)
        # args[3]: sampling_locations (Tensor)
        # args[4]: attention_weights (Tensor)
        # args[5]: im2col_step (int, e.g., 64)
        # Potentially other args like num_heads, num_levels, num_points if they are dynamic inputs
        # For this example, focusing on the 5 tensors + 1 int attribute from common implementations.

        if len(args) < 6:
            raise ValueError(
                f"Expected at least 6 arguments for {node_name}, got {len(args)}"
            )

        value_input             = args[0]
        value_spatial_shapes    = args[1]
        value_level_start_index = args[2]
        sampling_locations      = args[3]
        attention_weights       = args[4]
        im2col_step_val         = args[5] # This is the integer (e.g., 64)

        # Ensure im2col_step_val is an integer
        if not isinstance(im2col_step_val, int):
            raise TypeError(
                f"Expected im2col_step (arg 5) to be an int for {node_name}, "
                f"got {type(im2col_step_val)}"
            )

        # Cast shape-related tensors to INT64
        # ONNX DataType 7 is INT64. PyTorch's "Long" usually maps to INT64.
        # sym_help.cast_pytorch_to_onnx.get("Long", 7) would be robust if Long is always INT64 (it is)
        int64_onnx_type = getattr(torch.onnx.TensorProtoDataType, 'INT64', 7) # More robust way to get 7

        casted_spatial_shapes = g.op("Cast", value_spatial_shapes, to_i=int64_onnx_type)
        casted_level_start_index = g.op("Cast", value_level_start_index, to_i=int64_onnx_type)

        # Prepare the list of tensor inputs for the ONNX custom op
        plugin_tensor_inputs = [
            value_input,
            casted_spatial_shapes,
            casted_level_start_index,
            sampling_locations,
            attention_weights
        ]
        
        # Attributes for the custom op.
        # The name "im2col_step_i" (with _i suffix) is a convention for integer attributes
        # when using g.op(). The actual attribute name expected by your TensorRT plugin
        # might be different (e.g., "im2col_step", "im2col"). Consult your plugin's definition.
        # Let's assume the plugin expects an attribute named "im2col_step".
        return g.op("custom_op::MultiScaleDeformableAttentionPlugin",
                    *plugin_tensor_inputs,
                    im2col_step_i=im2col_step_val) # Pass as integer attribute
                    # If your plugin needs other attributes like num_heads, num_levels, num_points,
                    # and they were part of `*args` or `**kwargs` (or module attributes),
                    # extract them similarly and pass them as named attributes:
                    # num_heads_i=num_heads_val,
                    # num_levels_i=num_levels_val,

    # If node_name doesn't match, you might want to indicate it wasn't handled
    # or let the default ONNX exporter try (if this is part of a chain).
    # Depending on how `@register` works, returning None or raising an error might be options.
    # For now, implicitly returns None if not handled.
    return None # Or appropriate action if node_name doesn't match

def optimize():
    pass