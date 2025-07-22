# ComfyTP

Run diffusion models across multiple GPUs in ComfyUI using tensor parallelism!

## Current State:

✅ Ability to run larger models

✅ Memory distribution across GPUs

❌ Parallel computation speedup

## Testing Status:
Currently tested and seems to work with 2x3090s and 4x3090s. Generates images, but needs more testing, and need to figure out if we can get parallel processing going.

## Overview

This custom ComfyUI node enables you to split a single diffusion model across multiple GPUs, to do stuff that wouldn't fit on a single GPU or to distribute the computational load. This implements tensor parallelism, splitting the model layers across GPUs.

## Features

- Automatic model distribution based on parameter count
- Support for SD 1.5, SDXL, and other diffusion models (only SDXL tested so far)
- Easy to use with existing ComfyUI workflows
- No modifications needed to existing nodes

## Requirements

- ComfyUI
- PyTorch with CUDA support
- 2 or more NVIDIA GPUs
- Python 3.8+

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/M3T4L5T3F/ComfyTP
```

2. Restart ComfyUI

## Usage

1. Add the "Multi-GPU Diffusion Model" node to your workflow
2. Connect your model loader output to the node's input
3. Set the GPU IDs (e.g., "0,1" for first two GPUs, "0,1,2" for three GPUs)
4. Connect the output to your model node path to KSampler as usual

### Parameters

- **model**: The input diffusion model to distribute
- **gpu_ids**: Comma-separated list of GPU indices (e.g., "0,1" or "0,1,2,3")
- **verbose**: Enable detailed logging of model distribution and memory usage

### Example Workflow

```
[Checkpoint Loader] --> [Multi-GPU Diffusion Model] --> [KSampler] --> [VAE Decode] --> [Save Image]
                              |
                              gpu_ids: "0,1"
```

## How It Works

The node analyzes the model architecture and distributes layers across specified GPUs based on parameter count. During inference:

1. Input tensors start on the first GPU
2. Activations are moved between GPUs as they flow through the model
3. Skip connections are handled automatically
4. Output is returned to the original device for compatibility

## Performance Notes

- First generation will be slower due to initial setup
- Performance gains depend on model size and GPU communication bandwidth
- Best results with high-bandwidth GPU interconnects (NVLink, etc.)
- SDXL generation tested successfully on 2x RTX 3090 Ti

## Limitations

- It's pretty slow!
- Requires GPUs with similar memory capacity for balanced distribution
- Currently uses a simple parameter-based distribution strategy
- Additional memory overhead for cross-GPU communication

## Troubleshooting

If you encounter device mismatch errors:
- Ensure all specified GPUs are available
- Check that CUDA is properly initialized
- Try reducing batch size

## Contributing

Pull requests welcome! Some areas for improvement:
- Memory-aware distribution strategies
- Support for asymmetric GPU configurations
- Pipeline parallelism implementation
- Performance optimizations

## License

MIT

## Acknowledgments

Built for the ComfyUI community. Special thanks to the ComfyUI developers for creating such an extensible framework. Thanks to Claude for doing most of the heavy lifting! :p

---

If this helped you run larger models or improved your workflow, consider starring the repo!
