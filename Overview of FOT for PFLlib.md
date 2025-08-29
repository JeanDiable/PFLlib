### OT implementation report: orthogonal projection during aggregation and GPSE on task boundaries

- Setting: Standard cross-device FL with server-side model aggregation and client-side local SGD. FOT augments FedAvg with:

- FedProject: server applies an orthogonal projection to aggregated global updates to preserve old-task performance.

- GPSE: at task end, server privately extracts global principal subspaces for each layer and accumulates them into an orthonormal basis per layer.

### Notation and scope

- Let t denote the current task index; previous tasks are 1..t−1.

- For a given layer ℓ with parameter vector wℓ, let Uℓ ∈ ℝdℓ×kℓ be the accumulated global orthonormal basis spanning important directions of all previous tasks for that layer.

- Let gℓ be the server-side aggregated update for layer ℓ before projection; FOT applies a projection onto the orthogonal complement of span(Uℓ).

### Where it lives in the codebase

- Server aggregation with projection: FedML/fedml_api/distributed/fedavg_seq_cont/FedAVGAggregator.py, aggregate(...).

- Layer lists and state: constructor initializes self.orth_layer_names and self.orth_set (dict layer_name → basis).

- Projection application: inside aggregate, after forming global update, lines 215–224 apply the projection per layer (conv vs. non-conv shaping).

- Global principal subspace extraction (GPSE):

- Client-side activation collection: trainer/*_trainer.py, function collect_activations(...) for MNIST, CIFAR, ResNet, ImageNet variants.

- Server-side basis expansion: FedAVGAggregator.expand_orth_set(...).

### FedProject: projected aggregation (server)

- Server first computes the round’s aggregated parameters and an effective “global gradient”:

- Compute average over received client params; form gℓ as difference between previous global and averaged params.

- For each layer ℓ with existing basis Uℓ:

- If ℓ is convolutional or a residual shortcut, reshape gℓ to a 2D matrix by flattening spatial/filter dims as used in conv weights; otherwise keep 2D weight shape.

- Compute orthogonal projection Pℓ = Uℓ Uℓᵀ, and subtract Pℓ·gℓ from gℓ:

- conv/shortcut: project gℓ.view(out_channels, −1) along columns, then reshape back.

- dense: project gℓᵀ and transpose back.

- Update the global parameters by applying the projected update only. This enforces no-update along previously learned subspaces for old tasks.

- Layer coverage:

- MLP: ["layers.0.weight", "layers.3.weight", "layers.6.weight", "layers.9.weight"]

- AlexNet-like: ['conv1.weight','conv2.weight','conv3.weight','fc1.weight','fc2.weight']

- ResNet (CIFAR/ImageNet variants): extensive per-block conv and shortcut layers; lists are explicitly defined in the constructor.

Mathematically, with basis Uℓ ∈ ℝdℓ×kℓ, the projected update is

g̃ℓ = gℓ − Uℓ Uℓᵀ gℓ

applied with the correct flattening/unflattening for conv layers.

### GPSE: global principal subspace extraction (client + server)

- When a task ends (scheduler side), clients compute layer activations on local data and return only random-projected statistics. The server aggregates these (via secure aggregation primitives in the comm layer) and updates Uℓ via SVD.

Client side per layer ℓ (collect_activations):

- Build activation matrices:

- Conv layers: extract per-location patches via im2col/unfold into a matrix Aℓ ∈ ℝdℓ×nℓ, where dℓ = k_h·k_w·in_channels; nℓ is the number of extracted spatial positions × batch size.

- Dense layers or MLP: activations are transposed to Aℓ ∈ ℝdℓ×nℓ.

- For some ResNet/Imagenet variants, activations are zero-padded before unfold to align receptive fields.

- Privacy-preserving residualization:

- If a basis Uℓ exists, project away previous subspace: Rℓ = (I − Uℓ Uℓᵀ) Aℓ.

- Compute residual ratio rℓ = ||Rℓ||₂ / ||Aℓ||₂ to quantify remaining energy after projection.

- Random projection for privacy:

- Sample a client-side random Gaussian (or uniform) matrix Gℓ of appropriate shape, multiply Rℓ by Gℓ to form Yℓ = Rℓ Gℓ, and discard Rℓ and Gℓ.

- Return tuple [Yℓ, rℓ, bℓ] where bℓ is the number of local samples contributing.

- Shapes used in practice:

- CIFAR/ResNet/Imagenet conv: Yℓ ∈ ℝdℓ×mℓ where mℓ is typically proportional to dℓ (e.g., mℓ = 5·dℓ for some variants; see trainers for exact setting).

- MLP: random matrix is sampled in (−1, 1) with matching dimensions.

- Communication: the tuple per layer is included in the client runtime info structure and securely aggregated at the server (sum of Yℓ and sums of counts/ratios), without revealing raw Aℓ.

Server side basis expansion (expand_orth_set):

- Aggregate client tuples {Yℓ, rℓ, bℓ} across clients:

- Weighted residual ratio r̄ℓ = ∑i (bℓ,i / ∑j bℓ,j) · rℓ,i

- Adaptive energy threshold:

- Given current ε, compute a tightened target ε′ via ε′ = (r̄ℓ − (1 − ε)) / r̄ℓ.

- This increases required energy coverage if residual ratio is high, controlling how many new directions to add.

- SVD on aggregated projected statistics:

- Compute SVD of Yℓ to obtain left singular vectors Ûℓ and singular values Sℓ.

- Choose the smallest k such that cumulative energy ≥ ε′, i.e., ||Sℓ[1..k]||₂² / ||Sℓ||₂² ≥ ε′.

- Update basis:

- Concatenate Ûℓ[:, 1..k] to existing Uℓ (or initialize if none), then re-orthonormalize via QR to maintain orthonormal columns.

- Increase ε by eps_inc for subsequent tasks.

- Logging: logs subspace growth ratio per layer (basis width / dimension) to wandb.

### Data structures and scheduling

- orth_set: dict layer_name → torch.Tensor of shape (dℓ, kℓ).

- activation_dict: dict client_id → dict layer_name → [Yℓ, rℓ, bℓ].

- Scheduling:

- Normal rounds: server aggregates and applies projected updates using current orth_set.

- At task boundary: clients run collect_activations(...) and upload; server calls expand_orth_set() once activations are present, then clears activation_dict.

- Tests and logging orchestrated by server manager around task indices.

### Privacy properties

- Server never receives raw activations; it only sees sums of random projections Yℓ and aggregate ratios.

- Random matrices are sampled and kept on clients; secure aggregation hides individual client contributions.

- Residualization before projection prevents leakage of previously learned subspaces.

### Configuration knobs (hyperparameters)

- epsilon: target cumulative energy coverage per layer; determines basis growth.

- eps_inc: additive increment of ε after each expansion to gradually increase strictness.

- Model-specific orth_layer_names: which layers participate in projection/GPSE.

- Random projection width per layer: chosen in trainers; trade-off between estimation fidelity and bandwidth.

### Implementation nuances (shapes and conv handling)

- Conv layers:

- Projection in aggregation operates on gradients reshaped as [out_channels, −1]; the basis Uℓ is aligned to the flattened filter dimension dℓ.

- Activation collection uses torch.nn.Unfold (or explicit im2col loop in one ResNet trainer variant) with stride/padding consistent with the forward layer to produce Aℓ ∈ ℝdℓ×nℓ.

- Dense/MLP layers:

- Activations shaped as features×samples; projection is applied to transposed matrices where required.

### Migration guide: FedML → PFLlib

- Server-side aggregator hook:

- In PFLlib, implement a server aggregation callback that:

- Computes aggregated params or updates.

- Forms global gradient gℓ = wℓ(prev) − w̄ℓ(agg).

- Applies orthogonal projection g̃ℓ = (I − Uℓ Uℓᵀ) gℓ with layer-specific reshaping for conv vs dense.

- Updates global params wℓ(new) = wℓ(prev) − g̃ℓ.

- Maintain orth_set in server state; initialize from model-specific layer lists and persist across rounds/tasks.

- Client-side GPSE task-end job:

- Add a task-boundary job that:

- Collects layer activations on local train data for the current task.

- Residualizes against current Uℓ if present.

- Multiplies by a freshly sampled local random matrix Gℓ.

- Sends [Yℓ, rℓ, bℓ] to the server via PFLlib’s secure aggregation API.

- Server-side GPSE job:

- Upon receiving activations from all selected clients:

- Aggregate Yℓ, rℓ, bℓ across clients.

- Compute ε′ from weighted r̄ℓ and current ε.

- Run SVD on aggregated Yℓ and select components by cumulative energy ≥ ε′.

- Concatenate to Uℓ and re-orthonormalize (QR).

- Increase ε by eps_inc.

- Layer mapping:

- Define the same orth_layer_names for your model in PFLlib; ensure conv filter flatten dimension dℓ, unfold stride/pad, and shortcut layers match your model’s forward path.

- Minimal integration checklist:

- Server: projected-aggregation hook; persistent orth_set; conv vs dense reshaping; logging.

- Client: activation collector per model; residualization using provided Uℓ; random projection; secure aggregation envelope.

- Scheduler: trigger client GPSE at task boundary; call server basis expansion; clear buffers.

- Config: expose epsilon, eps_inc, layer list, projection widths.

### Assumptions and constraints

- Requires compatible layer naming or a mapping to PFLlib parameter keys.

- SVD on aggregated projected stats can be computed on CPU or GPU; ensure numerical stability and deterministic seeds if needed.

- Secure aggregation semantics must provide at least summation over tensors with consistent shapes across clients.

### References to concrete locations (for your review, not code replication)

- Server projection logic and orth_set lifecycle: FedML/fedml_api/distributed/fedavg_seq_cont/FedAVGAggregator.py, constructor, aggregate(...), expand_orth_set(...).

- Client activation builders:

- MNIST MLP: trainer/mnist_trainer.py::collect_activations

- CIFAR AlexNet-like: trainer/cifar_trainer.py::collect_activations

- ResNet (CIFAR): trainer/resnet_trainer.py::collect_activations

- ResNet/ImageNet: trainer/imagenet_trainer.py::collect_activations

—