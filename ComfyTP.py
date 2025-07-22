import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import comfy.model_management as mm
import comfy.utils
import comfy.model_base
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.ldm.modules.attention as attention
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
import folder_paths
import os
import json
import inspect

class ModelSplitter:
    """Handles splitting SD models across multiple GPUs"""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.layer_to_device = {}
        self.original_device_fn = None
        
    def analyze_model(self, model):
        """Analyze model structure and memory requirements"""
        total_params = 0
        layer_params = {}
        
        # Get the actual diffusion model
        if hasattr(model, 'model'):
            diffusion_model = model.model.diffusion_model
        else:
            diffusion_model = model.diffusion_model
            
        # Count parameters in each major block
        blocks = {
            'input_blocks': [],
            'middle_block': 0,
            'output_blocks': [],
            'out': 0,
            'time_embed': 0
        }
        
        # Input blocks (encoder)
        for i, block in enumerate(diffusion_model.input_blocks):
            params = sum(p.numel() for p in block.parameters())
            blocks['input_blocks'].append(params)
            total_params += params
            
        # Middle block
        params = sum(p.numel() for p in diffusion_model.middle_block.parameters())
        blocks['middle_block'] = params
        total_params += params
        
        # Output blocks (decoder)
        for i, block in enumerate(diffusion_model.output_blocks):
            params = sum(p.numel() for p in block.parameters())
            blocks['output_blocks'].append(params)
            total_params += params
            
        # Time embedding and output conv
        if hasattr(diffusion_model, 'time_embed'):
            params = sum(p.numel() for p in diffusion_model.time_embed.parameters())
            blocks['time_embed'] = params
            total_params += params
            
        if hasattr(diffusion_model, 'out'):
            params = sum(p.numel() for p in diffusion_model.out.parameters())
            blocks['out'] = params
            total_params += params
            
        return blocks, total_params
    
    def create_device_map(self, model) -> Dict[str, int]:
        """Create optimal device mapping for model layers"""
        blocks, total_params = self.analyze_model(model)
        
        # Get the actual diffusion model
        if hasattr(model, 'model'):
            diffusion_model = model.model.diffusion_model
        else:
            diffusion_model = model.diffusion_model
        
        device_map = {}
        
        # Create a list of all blocks with their parameter counts
        all_blocks = []
        
        # Add time embedding
        all_blocks.append(('time_embed', blocks['time_embed']))
        
        # Add input blocks
        for i in range(len(diffusion_model.input_blocks)):
            all_blocks.append((f'input_blocks.{i}', blocks['input_blocks'][i]))
        
        # Add middle block
        all_blocks.append(('middle_block', blocks['middle_block']))
        
        # Add output blocks
        for i in range(len(diffusion_model.output_blocks)):
            all_blocks.append((f'output_blocks.{i}', blocks['output_blocks'][i]))
        
        # Add final output
        all_blocks.append(('out', blocks['out']))
        
        # Distribute blocks across GPUs more evenly
        params_per_gpu = total_params / self.num_gpus
        current_gpu = 0
        current_gpu_params = 0
        
        # Always keep time_embed on first GPU for stability
        device_map['time_embed'] = self.gpu_ids[0]
        current_gpu_params = blocks['time_embed']
        
        # Distribute remaining blocks
        for i in range(1, len(all_blocks)):
            block_name, block_params = all_blocks[i]
            
            # Check if adding this block would exceed the target params per GPU
            # and we haven't reached the last GPU yet
            if (current_gpu_params + block_params > params_per_gpu * 1.2 and 
                current_gpu < self.num_gpus - 1):
                current_gpu += 1
                current_gpu_params = 0
            
            device_map[block_name] = self.gpu_ids[current_gpu]
            current_gpu_params += block_params
        
        # Ensure we're using all GPUs if possible
        # If we haven't used all GPUs, redistribute more aggressively
        if current_gpu < self.num_gpus - 1:
            print(f"Note: Only using {current_gpu + 1} out of {self.num_gpus} GPUs. Redistributing...")
            
            # Recalculate with more aggressive distribution
            blocks_per_gpu = len(all_blocks) / self.num_gpus
            for i, (block_name, _) in enumerate(all_blocks):
                if block_name == 'time_embed':
                    device_map[block_name] = self.gpu_ids[0]
                else:
                    target_gpu = min(int(i / blocks_per_gpu), self.num_gpus - 1)
                    device_map[block_name] = self.gpu_ids[target_gpu]
        
        return device_map

class TensorParallelDiffusionWrapper(nn.Module):
    """Wrapper that handles multi-GPU execution for diffusion models"""
    
    def __init__(self, model, device_map: Dict[str, int], gpu_ids: List[int]):
        super().__init__()
        self.model = model
        self.device_map = device_map
        self.gpu_ids = gpu_ids
        self.handles = []
        
        # Move layers to their assigned devices
        self._distribute_layers()
        
        # Verify the distribution
        self._verify_distribution()
        
    def _distribute_layers(self):
        """Move each layer to its assigned GPU"""
        diffusion_model = self.model.diffusion_model
        
        print("\n=== Distributing layers to GPUs ===")
        
        # Move each component to its assigned device
        for name, device_id in self.device_map.items():
            device = f'cuda:{device_id}'
            print(f"Moving {name} to {device}")
            
            if '.' in name:
                # Handle nested attributes like 'input_blocks.0'
                parts = name.split('.')
                component = diffusion_model
                parent = None
                last_part = None
                
                for i, part in enumerate(parts[:-1]):
                    parent = component
                    if part.isdigit():
                        component = component[int(part)]
                    else:
                        component = getattr(component, part)
                
                last_part = parts[-1]
                if last_part.isdigit():
                    # Get the module
                    module = component[int(last_part)]
                    # Use recursive move to ensure ALL submodules are moved
                    module = self._recursive_to_device(module, device)
                    # Re-assign the module
                    component[int(last_part)] = module
                else:
                    # Get the module
                    module = getattr(component, last_part)
                    # Use recursive move to ensure ALL submodules are moved
                    module = self._recursive_to_device(module, device)
                    # Re-assign the module
                    setattr(component, last_part, module)
            else:
                # Handle direct attributes
                if hasattr(diffusion_model, name):
                    module = getattr(diffusion_model, name)
                    module = self._recursive_to_device(module, device)
                    setattr(diffusion_model, name, module)
    
    def _recursive_to_device(self, module, device):
        """Recursively move all parameters and buffers to device"""
        # First, try to move the entire module using .to()
        # This should handle most cases correctly
        module = module.to(device)
        
        # Then, explicitly check and move any remaining parameters
        for name, param in module.named_parameters():
            if str(param.device) != device:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
        
        # Also check and move buffers
        for name, buffer in module.named_buffers():
            if str(buffer.device) != device:
                buffer.data = buffer.data.to(device)
                
        return module
    
    def _verify_distribution(self):
        """Verify that all layers are on their assigned devices"""
        diffusion_model = self.model.diffusion_model
        issues_found = False
        fixes_applied = 0
        
        print("\n=== Verifying device distribution ===")
        
        for name, expected_device_id in self.device_map.items():
            expected_device = f'cuda:{expected_device_id}'
            
            # Get the module
            if '.' in name:
                parts = name.split('.')
                module = diffusion_model
                parent = None
                last_part = None
                
                try:
                    for i, part in enumerate(parts):
                        parent = module
                        last_part = part
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                except Exception as e:
                    print(f"Warning: Could not access {name}: {e}")
                    continue
            else:
                if hasattr(diffusion_model, name):
                    module = getattr(diffusion_model, name)
                    parent = diffusion_model
                    last_part = name
                else:
                    continue
            
            # Check ALL parameters in the module
            wrong_params = []
            for param_name, param in module.named_parameters():
                if str(param.device) != expected_device:
                    wrong_params.append((param_name, str(param.device)))
                    
            if wrong_params:
                issues_found = True
                print(f"FIXING: {name} has {len(wrong_params)} parameters on wrong devices")
                for pname, pdevice in wrong_params[:3]:  # Show first 3
                    print(f"  - {pname}: on {pdevice}, expected {expected_device}")
                
                # Re-move the entire module
                module = self._recursive_to_device(module, expected_device)
                
                # Re-assign the module to ensure the parent reference is updated
                if parent is not None and last_part is not None:
                    if '.' in name:
                        parts = name.split('.')
                        component = diffusion_model
                        for part in parts[:-1]:
                            if part.isdigit():
                                component = component[int(part)]
                            else:
                                component = getattr(component, part)
                        
                        if parts[-1].isdigit():
                            component[int(parts[-1])] = module
                        else:
                            setattr(component, parts[-1], module)
                    else:
                        setattr(parent, last_part, module)
                
                fixes_applied += 1
                    
        if not issues_found:
            print("Device distribution verified successfully - all parameters on correct devices")
        else:
            print(f"Fixed {fixes_applied} misplaced modules")
            
            # Verify again to make sure fixes worked
            print("\n=== Re-verifying after fixes ===")
            still_wrong = False
            for name, expected_device_id in self.device_map.items():
                expected_device = f'cuda:{expected_device_id}'
                
                # Get the module again
                if '.' in name:
                    parts = name.split('.')
                    module = diffusion_model
                    try:
                        for part in parts:
                            if part.isdigit():
                                module = module[int(part)]
                            else:
                                module = getattr(module, part)
                    except:
                        continue
                else:
                    if hasattr(diffusion_model, name):
                        module = getattr(diffusion_model, name)
                    else:
                        continue
                
                # Quick check
                for param in module.parameters():
                    if str(param.device) != expected_device:
                        print(f"WARNING: {name} still has parameters on wrong device!")
                        still_wrong = True
                    break
                    
            if not still_wrong:
                print("All fixes successful!")
    
    def _check_device_placement(self):
        """Quick check that all modules are on their expected devices before forward pass"""
        diffusion_model = self.model.diffusion_model
        mismatches = []
        
        for name, expected_device_id in self.device_map.items():
            expected_device = f'cuda:{expected_device_id}'
            
            # Get the module
            if '.' in name:
                parts = name.split('.')
                module = diffusion_model
                try:
                    for part in parts:
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                except:
                    continue
            else:
                if hasattr(diffusion_model, name):
                    module = getattr(diffusion_model, name)
                else:
                    continue
            
            # Quick check - just look at first parameter
            for param in module.parameters():
                if str(param.device) != expected_device:
                    mismatches.append((name, str(param.device), expected_device))
                break
                
        if mismatches:
            print("\n!!! Device placement issues detected !!!")
            for name, actual, expected in mismatches:
                print(f"  {name}: on {actual}, expected {expected}")
            print("Attempting to fix...")
            self._verify_distribution()
    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """Forward pass with tensor movement between GPUs"""
        # Store the original input device - this is critical!
        input_device = x.device
        
        # Monitor GPU activity
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
            self._last_gpu_log = 0
            # Do a final device check on first forward
            self._check_device_placement()
            
        self._forward_count += 1
        
        # Log GPU usage periodically
        if self._forward_count - self._last_gpu_log >= 5:
            self._last_gpu_log = self._forward_count
            print(f"\n=== Forward Pass {self._forward_count} - GPU Activity ===")
            for gpu_id in self.gpu_ids:
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"GPU {gpu_id}: {allocated:.2f} GB / {total:.2f} GB allocated")
            print(f"Input shape: {x.shape}, Input device: {input_device}")
        
        # Get the dtype from the model
        diffusion_model = self.model.diffusion_model
        # Get model dtype from the first parameter we can find
        model_dtype = None
        for param in diffusion_model.parameters():
            model_dtype = param.dtype
            break
        if model_dtype is None:
            model_dtype = torch.float16  # fallback
        
        # Ensure initial tensors are on first GPU with correct dtype
        device = f'cuda:{self.gpu_ids[0]}'
        x = x.to(device=device, dtype=model_dtype)
        if timesteps is not None:
            # Keep timesteps as float32 for time embedding
            timesteps = timesteps.to(device=device, dtype=torch.float32)
        
        # Handle context properly - preserve its structure
        if context is not None:
            if isinstance(context, list):
                # Move each context element to device while preserving structure
                context_on_device = []
                for c in context:
                    if c is not None:
                        context_on_device.append(c.to(device=device, dtype=model_dtype))
                    else:
                        context_on_device.append(None)
                context = context_on_device
            else:
                context = context.to(device=device, dtype=model_dtype)
                
        if y is not None:
            y = y.to(device=device, dtype=model_dtype)
            
        # Get timestep embedding
        diffusion_model = self.model.diffusion_model
        
        # Debug timestep shape
        if self._forward_count == 1:
            print(f"Timestep shape: {timesteps.shape}, dtype: {timesteps.dtype}")
            print(f"X shape: {x.shape}, dtype: {x.dtype}")
            
            # Check the time_embed structure
            print(f"Time embed type: {type(diffusion_model.time_embed)}")
            if hasattr(diffusion_model.time_embed, '__len__'):
                print(f"Time embed modules: {len(diffusion_model.time_embed)}")
            
            # Check what dtype the first layer of time_embed expects
            if hasattr(diffusion_model.time_embed, '0') and hasattr(diffusion_model.time_embed[0], 'weight'):
                expected_dtype = diffusion_model.time_embed[0].weight.dtype
                print(f"Time embed first layer weight dtype: {expected_dtype}")
        
        # Process timesteps to ensure correct shape
        # Most SD models expect timesteps to be shape [batch_size]
        if timesteps.dim() == 0:
            # Single timestep scalar
            timesteps = timesteps.unsqueeze(0).expand(x.shape[0])
        elif timesteps.dim() == 2:
            # If we get [batch, 1] or [1, batch], flatten to [batch]
            timesteps = timesteps.squeeze()
            if timesteps.dim() == 0:
                timesteps = timesteps.unsqueeze(0)
        
        # Ensure we have the right batch size
        if timesteps.shape[0] == 1 and x.shape[0] > 1:
            timesteps = timesteps.expand(x.shape[0])
        elif timesteps.shape[0] != x.shape[0]:
            print(f"WARNING: timestep batch size {timesteps.shape[0]} doesn't match x batch size {x.shape[0]}")
        
        # IMPORTANT: Convert timesteps to model dtype to match the linear layer weights
        # The time_embed layer has float16 weights, so we need float16 inputs
        timesteps = timesteps.to(dtype=model_dtype)
        
        if self._forward_count == 1:
            print(f"Final timestep shape for time_embed: {timesteps.shape}, dtype: {timesteps.dtype}")
        
        # Check if this is an SDXL model (has label_emb)
        if hasattr(diffusion_model, 'label_emb'):
            # SDXL models need special handling with timestep embeddings
            # Detect the expected input dimension from the first layer of time_embed
            if hasattr(diffusion_model.time_embed, '0') and hasattr(diffusion_model.time_embed[0], 'in_features'):
                time_embed_dim = diffusion_model.time_embed[0].in_features
            else:
                # Default to 320 for SDXL if we can't detect
                time_embed_dim = 320
            
            if self._forward_count == 1:
                print(f"Detected SDXL model with time embedding dimension: {time_embed_dim}")
            
            # Create timestep embeddings with the correct dimension
            t_emb = diffusion_model.time_embed(timestep_embedding(timesteps, time_embed_dim).to(dtype=model_dtype))
            
            # Create a vector_embeds tensor if not provided
            if y is None:
                vector_embeds = torch.zeros((x.shape[0], diffusion_model.label_emb.num_embeddings), device=x.device, dtype=model_dtype)
            else:
                vector_embeds = y
                
            # Add label embeddings if present
            if diffusion_model.label_emb is not None:
                emb = diffusion_model.label_emb(vector_embeds)
                t_emb = t_emb + emb
        else:
            # Regular SD 1.5 style models - pass timesteps directly
            t_emb = diffusion_model.time_embed(timesteps)
        
        # Store intermediate activations
        h = x
        hs = []
        
        # Debug context structure
        if self._forward_count == 1 and context is not None:
            print(f"\n=== Context Structure Debug ===")
            if isinstance(context, list):
                print(f"Context is a list with {len(context)} elements")
                for i, ctx in enumerate(context):
                    if ctx is not None:
                        print(f"  Context[{i}]: shape {ctx.shape}, dtype {ctx.dtype}")
                    else:
                        print(f"  Context[{i}]: None")
            else:
                print(f"Context is a tensor: shape {context.shape}, dtype {context.dtype}")
        
        # Input blocks
        for i, module in enumerate(diffusion_model.input_blocks):
            device = f'cuda:{self.device_map.get(f"input_blocks.{i}", self.gpu_ids[0])}'
            h = h.to(device=device, dtype=model_dtype)
            # Keep t_emb in its original dtype (might be different from model dtype)
            t_emb_device = t_emb.to(device=device)
            
            # Debug for the problematic block
            if i == 1 and self._forward_count == 1:
                print(f"\n=== Debug at input_blocks[{i}] ===")
                print(f"Module type: {type(module)}")
                print(f"H shape before: {h.shape}")
                if hasattr(module, '1') and hasattr(module[1], 'transformer_blocks'):
                    print(f"Has transformer blocks: Yes")
            
            # Move context to device
            if context is not None:
                if isinstance(context, list):
                    context_device = [c.to(device=device, dtype=model_dtype) if c is not None else None for c in context]
                else:
                    context_device = context.to(device=device, dtype=model_dtype)
            else:
                context_device = None
            
            # Move transformer_options to device if needed
            transformer_options = kwargs.get('transformer_options', {})
            
            # Handle different block types
            if hasattr(module, 'forward'):
                # Special handling for TimestepEmbedSequential modules
                if type(module).__name__ == 'TimestepEmbedSequential':
                    # These modules forward context through their internal layers
                    try:
                        h = module(h, t_emb_device, context=context_device, transformer_options=transformer_options)
                    except TypeError as e:
                        # Some modules might not accept all arguments
                        if "got an unexpected keyword argument 'context'" in str(e):
                            h = module(h, t_emb_device)
                        elif "got an unexpected keyword argument 'transformer_options'" in str(e):
                            h = module(h, t_emb_device, context=context_device)
                        else:
                            raise
                else:
                    # For other module types
                    try:
                        sig = inspect.signature(module.forward)
                        params = list(sig.parameters.keys())
                        
                        if 'context' in params and context_device is not None:
                            h = module(h, t_emb_device, context=context_device)
                        elif 'emb' in params or 'timestep' in params or len(params) >= 2:
                            h = module(h, t_emb_device)
                        else:
                            h = module(h)
                    except Exception as e:
                        print(f"Error in input_blocks[{i}]: {e}")
                        print(f"Module type: {type(module)}")
                        print(f"H shape: {h.shape}, t_emb shape: {t_emb_device.shape}")
                        if context_device is not None:
                            if isinstance(context_device, list):
                                print(f"Context is list with {len(context_device)} elements")
                            else:
                                print(f"Context shape: {context_device.shape}")
                        raise
            else:
                h = module(h)
                
            hs.append(h)
        
        # Middle block
        device = f'cuda:{self.device_map.get("middle_block", self.gpu_ids[0])}'
        h = h.to(device=device, dtype=model_dtype)
        # Keep t_emb in its original dtype
        t_emb_device = t_emb.to(device=device)
        
        # Debug device issues
        if self._forward_count == 1:
            print(f"\n=== Middle Block Debug ===")
            print(f"Target device: {device}")
            print(f"H device: {h.device}")
            print(f"t_emb device: {t_emb_device.device}")
            
            # Check middle block parameters
            if hasattr(diffusion_model.middle_block, 'parameters'):
                param_devices = set()
                for name, p in diffusion_model.middle_block.named_parameters():
                    param_devices.add(str(p.device))
                    if self._forward_count == 1 and len(param_devices) > 1:
                        print(f"  Parameter {name}: {p.device}")
                print(f"Middle block parameter devices: {param_devices}")
        
        # Handle context for middle block
        if context is not None:
            if isinstance(context, list):
                context_device = [c.to(device=device, dtype=model_dtype) if c is not None else None for c in context]
            else:
                context_device = context.to(device=device, dtype=model_dtype)
        else:
            context_device = None
        
        # Move transformer_options
        transformer_options = kwargs.get('transformer_options', {})
        
        try:
            h = diffusion_model.middle_block(h, t_emb_device, context=context_device, transformer_options=transformer_options)
        except TypeError as e:
            if "got an unexpected keyword argument 'transformer_options'" in str(e):
                h = diffusion_model.middle_block(h, t_emb_device, context=context_device)
            else:
                raise
        
        # Output blocks
        for i, module in enumerate(diffusion_model.output_blocks):
            device = f'cuda:{self.device_map.get(f"output_blocks.{i}", self.gpu_ids[-1])}'
            
            # Debug device tracking
            if self._forward_count <= 2:
                print(f"\n=== Processing output_blocks[{i}] ===")
                print(f"Target device: {device}")
                print(f"H device before move: {h.device}")
                print(f"Skip connection device: {hs[-1].device}")
            
            h = h.to(device=device, dtype=model_dtype)
            h_skip = hs.pop().to(device=device, dtype=model_dtype)
            # Keep t_emb in its original dtype
            t_emb_device = t_emb.to(device=device)
            
            if self._forward_count <= 2:
                print(f"H device after move: {h.device}")
                print(f"h_skip device after move: {h_skip.device}")
            
            # Check module parameters are on correct device
            if self._forward_count == 1 and hasattr(module, 'parameters'):
                param_devices = set()
                for p in module.parameters():
                    param_devices.add(p.device)
                if len(param_devices) > 1:
                    print(f"WARNING: Mixed devices in output_blocks[{i}]: {param_devices}")
                elif len(param_devices) == 1:
                    expected_device = next(iter(param_devices))
                    if h.device != expected_device:
                        print(f"WARNING: Tensor device mismatch in output_blocks[{i}]:")
                        print(f"  Tensor on: {h.device}")
                        print(f"  Module expects: {expected_device}")
            
            # Handle context for output blocks
            if context is not None:
                if isinstance(context, list):
                    context_device = [c.to(device=device, dtype=model_dtype) if c is not None else None for c in context]
                else:
                    context_device = context.to(device=device, dtype=model_dtype)
            else:
                context_device = None
            
            # Move transformer_options
            transformer_options = kwargs.get('transformer_options', {})
            
            # Concatenate skip connection
            h = torch.cat([h, h_skip], dim=1)
            
            # Forward through block
            if hasattr(module, 'forward'):
                # Special handling for TimestepEmbedSequential modules
                if type(module).__name__ == 'TimestepEmbedSequential':
                    try:
                        h = module(h, t_emb_device, context=context_device, transformer_options=transformer_options)
                    except TypeError as e:
                        if "got an unexpected keyword argument 'context'" in str(e):
                            h = module(h, t_emb_device)
                        elif "got an unexpected keyword argument 'transformer_options'" in str(e):
                            h = module(h, t_emb_device, context=context_device)
                        else:
                            raise
                    except RuntimeError as e:
                        print(f"RuntimeError in output_blocks[{i}]: {e}")
                        print(f"Module device check:")
                        for name, param in module.named_parameters():
                            print(f"  {name}: {param.device}")
                            break  # Just show first few
                        raise
                else:
                    try:
                        sig = inspect.signature(module.forward)
                        params = list(sig.parameters.keys())
                        
                        if 'context' in params and context_device is not None:
                            h = module(h, t_emb_device, context=context_device)
                        elif 'emb' in params or 'timestep' in params or len(params) >= 2:
                            h = module(h, t_emb_device)
                        else:
                            h = module(h)
                    except Exception as e:
                        print(f"Error in output_blocks[{i}]: {e}")
                        raise
            else:
                h = module(h)
        
        # Final output
        device = f'cuda:{self.device_map.get("out", self.gpu_ids[-1])}'
        h = h.to(device)
        h = diffusion_model.out(h)
        
        # CRITICAL: Move output back to the original input device!
        # This ensures the denoising calculation works correctly
        h = h.to(input_device)
        
        if self._forward_count <= 2:
            print(f"\n=== Output Debug ===")
            print(f"Final output device: {h.device}")
            print(f"Original input device: {input_device}")
        
        return h

class MultiGPUDiffusionModel:
    """ComfyUI node for multi-GPU model execution"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "gpu_ids": ("STRING", {
                    "default": "0,1", 
                    "multiline": False,
                    "tooltip": "Comma-separated GPU IDs (e.g., '0,1,2')"
                }),
                "verbose": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "split_model"
    CATEGORY = "advanced/model"
    
    def split_model(self, model, gpu_ids, verbose):
        """Split model across multiple GPUs"""
        
        # Parse GPU IDs
        gpu_list = [int(x.strip()) for x in gpu_ids.split(',')]
        
        # Validate GPUs
        available_gpus = torch.cuda.device_count()
        
        if verbose:
            print(f"\n=== GPU Detection ===")
            print(f"Available GPUs: {available_gpus}")
            for i in range(available_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        for gpu_id in gpu_list:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available. Found {available_gpus} GPUs.")
        
        if len(gpu_list) < 2:
            print("Warning: Less than 2 GPUs specified. Model will not be split.")
            print("Returning original model without modification.")
            return (model,)
        
        # Create model splitter
        splitter = ModelSplitter(gpu_list)
        
        # Analyze model
        blocks, total_params = splitter.analyze_model(model)
        
        if verbose:
            print(f"\n=== Model Analysis ===")
            print(f"Total parameters: {total_params / 1e9:.2f}B")
            print(f"Input blocks: {len(blocks['input_blocks'])}")
            print(f"Output blocks: {len(blocks['output_blocks'])}")
        
        # Create device map
        device_map = splitter.create_device_map(model)
        
        if verbose:
            print(f"\n=== Device Mapping ===")
            gpu_assignments = {}
            for layer, gpu_id in device_map.items():
                if gpu_id not in gpu_assignments:
                    gpu_assignments[gpu_id] = []
                gpu_assignments[gpu_id].append(layer)
            
            for gpu_id in gpu_list:
                if gpu_id in gpu_assignments:
                    print(f"GPU {gpu_id}: {len(gpu_assignments[gpu_id])} components")
                    # Show first few components for verification
                    for comp in gpu_assignments[gpu_id][:3]:
                        print(f"  - {comp}")
                    if len(gpu_assignments[gpu_id]) > 3:
                        print(f"  ... and {len(gpu_assignments[gpu_id]) - 3} more")
                else:
                    print(f"GPU {gpu_id}: No components assigned")
            
            # Warn if not all GPUs are used
            used_gpus = len(gpu_assignments)
            if used_gpus < len(gpu_list):
                print(f"\nWARNING: Only {used_gpus} out of {len(gpu_list)} GPUs are being utilized")
                print("Consider using a larger model or adjusting distribution strategy")
        
        # Clone the model to avoid modifying the original
        cloned_model = model.clone()
        
        # Replace the forward method with our multi-GPU version
        original_model = cloned_model.model
        wrapped_model = TensorParallelDiffusionWrapper(original_model, device_map, gpu_list)
        
        # Create a more comprehensive wrapper
        class MultiGPUModelWrapper(nn.Module):
            def __init__(self, wrapped_model, original_model):
                super().__init__()
                self.wrapped_diffusion_model = wrapped_model
                self.original_model = original_model
                
                # CRITICAL: Replace the actual diffusion_model's forward method
                # This ensures any direct calls to diffusion_model() go through our multi-GPU path
                self.original_forward = original_model.diffusion_model.forward
                original_model.diffusion_model.forward = self._multi_gpu_forward
                
                # Also set our wrapped model as the diffusion_model
                self.diffusion_model = original_model.diffusion_model
                
                # Copy all other attributes
                for attr in dir(original_model):
                    if not attr.startswith('_') and attr not in ['diffusion_model', 'forward']:
                        try:
                            setattr(self, attr, getattr(original_model, attr))
                        except:
                            pass
            
            def _multi_gpu_forward(self, x, timesteps, context=None, y=None, **kwargs):
                """Direct replacement for diffusion_model.forward"""
                print(f"\n=== Multi-GPU Forward Called! Shape: {x.shape} ===")
                # Route through our tensor parallel wrapper
                return self.wrapped_diffusion_model(x, timesteps, context=context, y=y, **kwargs)
            
            def forward(self, *args, **kwargs):
                # Fallback forward method
                return self._multi_gpu_forward(*args, **kwargs)
            
            def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
                print("\n=== apply_model called ===")
                sigma = t
                xc = x
                context = c_crossattn
                y = kwargs.get('y', None)
                
                # Call our multi-GPU forward
                return self._multi_gpu_forward(xc, sigma, context=context, y=y, **kwargs)
        
        cloned_model.model = MultiGPUModelWrapper(wrapped_model, original_model)
        
        # Print memory usage
        if verbose:
            print(f"\n=== GPU Memory Status ===")
            for gpu_id in gpu_list:
                torch.cuda.set_device(gpu_id)
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                total_mem = torch.cuda.mem_get_info()[1] / 1024**3
                used_mem = total_mem - free_mem
                print(f"GPU {gpu_id}: {used_mem:.1f}/{total_mem:.1f} GB used")
        
        return (cloned_model,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "MultiGPUDiffusionModel": MultiGPUDiffusionModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiGPUDiffusionModel": "Multi-GPU Diffusion Model",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
