# 🧪 `ik_llama.cpp` — Backend gRPC for LocalAI (CPU-Optimized)

> **Status:** *Alpha* — Functional in production for basic inference, under active development.  
> **Base version:** `ik_llama.cpp` v5429887 (custom fork for educational AI systems)  
> **Focus:** CPU-only inference with OpenBLAS, NUMA, AVX2 optimized for Intel Xeon servers (dual-socket)

A gRPC backend for [LocalAI](https://github.com/mudler/LocalAI), built on `llama.cpp` and extended as **`ik_llama.cpp`** — designed for educational environments requiring full privacy, local control, and performance on on-premise infrastructure (e.g., the **Cuyex** system at *Colegio Santa Rosa de Lima*).

---

## ✅ Features

| Functionality | Status | Notes |
|---------------|--------|-------|
| **Chat / Completions** | ✅ Stable | Streaming, tools, multimodal (image/audio) |
| **Embeddings** | ⚠️ Alpha | Basic support; pooling and normalization under active development |
| **Reranking** | ✅ Stable | For RAG and semantic search |
| **Tools & Grammar** | ✅ Stable | `tools`, `tool_choice`, `grammar`, `json_schema` |
| **LoRA & RoPE Scaling** | ✅ Stable | YARN, Linear, Flash Attention |
| **NUMA (dual-socket)** | ✅ Stable | Automatic distribution across dual-socket Xeon |

### ⚙️ Compilation

The only supported target is:

| Target | Architecture | Recommended for |
|--------|--------------|------------------|
| `llama-cpp-grpc` | x86_64 with AVX2 (and optionally AVX512) | ✅ **Only supported target** |

> 🔑 This target compiles `grpc-server` as a static binary (no runtime dependencies), ready to be registered in LocalAI.  
> The resulting binary is optimized with **AVX2**, **FMA**, **BMI2**, **OpenBLAS**, and **LTO**, and is compatible with **Broadwell-EP and newer** architectures.

---

## 📦 Installation (LocalAI GUI)

> 💡 This backend is installed **manually** via the LocalAI web interface. No system rebuild required.

### Step 1: Compile the backend
```bash
# Clone or copy this repo into your development environment
cd /path/to/your/local-repo/ik_llama-cpp

# Compile using the only supported target
make llama-cpp-grpc

# Verify the static binary was generated
ls -lh grpc-server
# → Should show something like: -rwxr-xr-x 1 root root 180M ...
```

> ⚠️ **Important**: The functional binary is `grpc-server`. The `llama-cpp-grpc` target is the only supported one.

### Step 2: Copy to LocalAI backend directory
```bash
# Create the backend directory (fixed name: `cpu-ikllama-cpp`)
sudo mkdir -p /opt/local-path-backends/cpu-ikllama-cpp

# Copy the main binary
sudo cp grpc-server /opt/local-path-backends/cpu-ikllama-cpp/

# Copy dynamic dependencies (from existing `cpu-llama-cpp`)
sudo cp -r /opt/local-path-backends/cpu-llama-cpp/lib/* /opt/local-path-backends/cpu-ikllama-cpp/lib/

# Verify
ls /opt/local-path-backends/cpu-ikllama-cpp/
# → Should show: grpc-server  lib/
```

### Step 3: Register in LocalAI GUI
1. Open LocalAI web interface (`http://localhost:8080`)
2. Go to **Backend Management** → **Manual Install**
3. Fill in:
   - **OCI Image / URL / Path**: `/opt/local-path-backends/cpu-ikllama-cpp/grpc-server`
   - **Name (required)**: `cpu-ikllama-cpp`
   - **Alias (optional)**: `ikllama`
4. Click **Install**

✅ The backend will now be available for model assignment.

---

## 📄 Model Definition Example

Create a YAML file in `/opt/local-path-provisioner/` (or use the API to load models):

```yaml
# /opt/local-path-provisioner/qwen3.5-397b.yaml
backend: cpu-ikllama-cpp
context_size: 262144
f16: false
mmap: true
mmlock: false
function:
    grammar:
        disable: true
known_usecases:
    - chat
name: Qwen3.5-397B-A17B
options:
    - use_jinja:false
    - cache_ram:-1
    - cache_reuse:262144
    - context_shift:false
    - swa_full:false
    - no_op_offload:true
    - cont_batching:true
    - sps:0.05
    - kv_unified:true
    - n_ubatch:2048
    - n_threads_batch:28
    - attn_max_batch:4096
    - ctx_checkpoints:32
parameters:
    model: Qwen3.5-397B-A17B-UD-IQ3_XXS_IK.gguf
    model_base_name: qwen3.5-397b
    batch: 8192
    mirostat: 0
    temperature: 0.7
    top_p: 0.8
    top_k: 20
    presence_penalty: 1.5
    keep: -1
    cache_prompt: true
reasoning:
    disable: true
numa: false
flash_attention: on
prompt_cache_path: "cache/qwen3-397B"
prompt_cache_all: true
prompt_cache_ro: false
cache_type_k: q8_0
cache_type_v: q8_0
template:
    use_tokenizer_template: true
```

---

## 🖼️ Functional Testing

### ✅ CuyexLLM (AnythingLLM for Kubernetes, CPU-only)
<img src="https://github.com/user-attachments/assets/7d21a312-6be9-45f1-b1df-973bdbb62f02" alt="CuyexLLM inference test with ik_llama.cpp on Xeon E5-2690v4" width="100%" />

*Inference running in CuyexLLM (fork of AnythingLLM optimized for Kubernetes with CPU-only), using `grpc-server` compiled with AVX2 on an Intel Xeon E5-2690v4 server (dual socket, 14 nm, Broadwell-EP).*

---

### ✅ LocalAI with `cpu-ikllama-cpp` registered
<img src="https://github.com/user-attachments/assets/3002383e-eaf0-4632-bcb6-24edbcff46d2" alt="LocalAI backend registration and model loading on Xeon E5-2690v4" width="100%" />

*Registration of the `cpu-ikllama-cpp` backend in LocalAI's web interface and successful loading of the Qwen3.5-397B model. Tested on real hardware: Intel Xeon E5-2690v4 (dual socket), AVX2 enabled.*

---

## ⚠️ Known Limitations (Alpha)

| Functionality | Status | Notes |
|---------------|--------|-------|
| **Embeddings** | ⚠️ Partial | Basic support only; pooling and normalization under active development |
| **Advanced multimodal** | ⚠️ Limited | Single image/audio per prompt; multi-media scenarios not yet exhaustively tested |
| **GPU** | ❌ Not included | CPU-only by design; would require rebuild with `-DGGML_CUDA=ON` |

---

## 🏗️ Technical Details

- **Language**: C++20 (gRPC, Protobuf, `absl`, `llama.cpp`)
- **Build**: CMake 3.15+ with static gRPC integration
- **Protocol**: gRPC (proto located in `LocalAI/backend/backend.proto`)
- **Supported formats**: GGUF, GGML, LoRA, MMProj (multimodal)

### Critical patches applied
- `patches/iqk_fa_s_zero.patch`: Prevents crash on empty slots during parallel inference (essential for multi-user educational environments)

---

## 📚 Why This Backend?

This project addresses specific needs in educational environments:
- ✅ **Full privacy**: All processing happens locally  
- ✅ **Zero cost**: No dependency on external APIs  
- ✅ **Scalable**: Optimized for dedicated servers (dual-socket Xeon)  
- ✅ **Maintainable**: Static build, no runtime dependencies  

Part of the **Cuyex** system, developed at *Colegio Santa Rosa de Lima* to democratize AI access in public schools.

---

## 📜 License

This project is licensed under the **Apache-2.0 License** — see the [LICENSE](LICENSE) file for details.  
Based on [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (MIT) and [`LocalAI`](https://github.com/mudler/LocalAI) (Apache-2.0).

---

## 🤝 Want to Contribute?

Help improve this backend! Priority areas:
- Robust embeddings validation (pooling, normalization)  
- Advanced multimodal testing (multiple images/audio)  
- Technical documentation (benchmarking, troubleshooting)  
- Optimizations for new models (Qwen3, Llama 4, etc.)
