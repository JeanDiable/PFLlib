### PFLlib-aligned migration mapping for FOT (no code)

- Target repo: https://github.com/TsingZ0/PFLlib.git

- Execution loop to hook:

- Server loop: system/flcore/servers/serveravg.py::FedAvg.train()

- Base aggregation: system/flcore/servers/serverbase.py::Server.aggregate_parameters()

- Client loop: system/flcore/clients/clientavg.py::clientAVG.train() and clientbase.Client

### Where FedProject belongs (server projected aggregation)

- Insert an orthogonal-projection step between receive and final param update:

- After receive_models() and before/within aggregate_parameters().

- Compute averaged client params (already done via add_parameters), then compute per-layer effective global update gℓ = wℓ(prev) − w̄ℓ(agg).

- Apply g̃ℓ = (I − Uℓ Uℓᵀ) gℓ using per-layer basis Uℓ in server state.

- Set new global params wℓ(new) = wℓ(prev) − g̃ℓ.

- PFLlib API equivalents:

- Server.global_model: torch.nn.Module holding current server weights.

- Server.uploaded_models: list of client models.

- Server.uploaded_weights: normalized weights for aggregation.

- State to add in Server:

- orth_set: dict layer_name → tensor basis per layer.

- epsilon, eps_inc.

- orth_layer_names constructed from args.model architecture (map to parameter names in PFLlib models, e.g., ResNet, AlexNet).

- Shape handling:

- Conv/shortcut params: flatten kernel dims to dℓ, project with Uℓ Uℓᵀ along flattened dimension; reshape back.

- Dense/MLP: 2-D weight matrices; project either on param or transposed param as in FedML.

### Where GPSE belongs (client task-end activation collection + server basis update)

- Client additions:

- A task-boundary job that:

- Loads local train data via clientbase.Client.load_train_data().

- Builds per-layer activation matrices Aℓ:

- Conv: im2col via torch.nn.Unfold (match kernel, stride, pad of layer), produce Aℓ ∈ ℝdℓ×nℓ.

- Dense/MLP: feature×sample matrices.

- Optional padding for ResNet-like to align receptive field as done in FedML trainers.

- Residualize using current Uℓ (downloaded or provided via control message from server):

- Rℓ = (I − Uℓ Uℓᵀ) Aℓ; compute residual ratio rℓ = ||Rℓ||₂ / ||Aℓ||₂.

- Random projection: sample local Gaussian matrix Gℓ, produce Yℓ = Rℓ Gℓ; send [Yℓ, rℓ, bℓ] to server via standard client-to-server payload extension.

- Placement: extend clientAVG.train() to detect task end (round index from server) and trigger activation collection once.

- Server additions:

- Buffer activation_dict: client_id → {layer_name: [Yℓ, rℓ, bℓ]}.

- On task boundary (e.g., every time you declare a task end in the scheduler), after all clients uploaded activations:

- Aggregate per layer across clients: sum Yℓ and weighted stats by bℓ.

- Compute ε′ from weighted residual r̄ℓ and current ε.

- Run SVD on aggregated Yℓ, pick top-k by cumulative energy ≥ ε′.

- Update orth_set[layer] by concatenation then QR re-orthonormalization.

- Increment ε by eps_inc, clear buffers.

### Scheduling and signaling

- PFLlib round loop resides in FedAvg.train(); define task boundaries (e.g., specific round indices).

- Add a simple protocol flag in server to signal clients at task end:

- Option A: overload send_models() to also send meta flags and orth_set headers.

- Option B: clients compute activations using latest orth_set fetched at beginning of boundary round.

### Model layer naming alignment

- PFLlib’s models in system/flcore/trainmodel/:

- ResNet: resnet.py or torchvision; map parameter names to FOT layer lists (conv layers and shortcuts).

- AlexNet: alexnet.py param names for conv/fc.

- MLP/CNN: models.py for DNN/CNN; map linear/conv names similarly.

- Build orth_layer_names by inspecting state_dict().keys() of args.model, choose layers that correspond to trainable conv/fc params and skip BN.

### Privacy and transport in PFLlib

- Use existing message pathways:

- Server.receive_models() currently collects models; extend to also collect per-client “activation payloads” carried by clients after boundary.

- Client.save_item/load_item can persist random matrices locally if needed; do not transmit them.

- Ensure only Yℓ, rℓ, bℓ are transmitted; raw activations and Gℓ remain client-local. If secure aggregation is desired, wrap the per-layer tensors using any available secure-sum mode; if not provided by PFLlib, implement sum over tensors but rely on trust for now.

### Minimal variable/state additions

- On server:

- self.orth_set: Dict[str, Tensor]

- self.epsilon: float, self.eps_inc: float

- self.activation_dict: Dict[int, Dict[str, Tuple[Tensor, float, int]]]

- self.orth_layer_names: List[str]

- On client:

- Access to orth_set snapshots for residualization.

- One-shot activation collection routine with model-aware im2col.

### Round-level control flow in PFLlib terms

- Each round:

- Server select/send → clients train → server receive → server aggregate.

- If boundary:

- Clients: run activation collection and upload Yℓ,rℓ,bℓ payloads.

- Server: run basis expansion from payloads; clear activation buffer.

- Next rounds use updated orth_set to project updates.

### Hyperparameters surface in PFLlib

- Add to system/main.py parser:

- --epsilon, --eps_inc

- --fot_layers (optional preset selector)

- --task_schedule (boundary indices)

- --gpse_proj_width_factor per layer type

- Store in args; initialize in Server.__init__.

### Validation checklist

- Param name mapping verified against args.model.state_dict().keys().

- Conv flatten dimension dℓ consistent with im2col used in training and activation collection.

- Projection math matches FedML (transpose handling).

- Basis QR re-orthonormalization after concatenation.

- Logging: basis growth over rounds; residual ratios.

If you want, I can add a concise mapping table layer-by-layer for your exact chosen model in PFLlib.

- Completed: extracted FOT internals and drafted PFLlib migration mapping.

- Impact: you can implement FOT in PFLlib by adding server state/hooks and a task-end client job.