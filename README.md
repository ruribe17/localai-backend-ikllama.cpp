# 🧪 `ik_llama.cpp` — Backend gRPC para LocalAI (CPU-Optimized)

> **Estado:** *Alpha* — Funcional en producción para inferencia básica, en desarrollo activo.  
> **Versión base:** `ik_llama.cpp` v5429887 (fork personalizado para sistemas educativos)  
> **Enfoque:** Inferencia en CPU con OpenBLAS, NUMA, AVX2/AVX512 optimizado para servidores Intel Xeon (dual-socket)

Backend gRPC compatible con [LocalAI](https://github.com/mudler/LocalAI), construido sobre `llama.cpp` y extendido como **`ik_llama.cpp`** — diseñado para entornos educativos que requieren privacidad, control total y rendimiento en infraestructura local (como el sistema **Cuyex** del *Colegio Santa Rosa de Lima*).

---

## ✅ Características

| Funcionalidad | Estado | Notas |
|---------------|--------|-------|
| **Chat / Completions** | ✅ Estable | Streaming, herramientas, multimodal (imagen/audio) |
| **Embeddings** | ⚠️ Alpha | Básico; pooling y normalización en desarrollo |
| **Reranking** | ✅ Estable | Para RAG y búsqueda semántica |
| **Tools & Grammar** | ✅ Estable | `tools`, `tool_choice`, `grammar`, `json_schema` |
| **LoRA & RoPE Scaling** | ✅ Estable | YARN, Linear, Flash Attention |
| **NUMA (dual-socket)** | ✅ Estable | Distribución automática en Xeon dual socket |

### ⚙️ Compilación
El único target funcional es:

| Target | Arquitectura | Recomendado para |
|--------|--------------|------------------|
| `llama-cpp-grpc` | x86_64 con soporte AVX2 | ✅ **Único target soportado** |

> 🔑 Este target compila `grpc-server` como binario estático (sin dependencias en runtime), listo para ser registrado en LocalAI.  
> Otras opciones (como `llama-cpp-avx2`) **no están soportadas** y pueden causar inestabilidad.

🔑 Usa OpenBLAS, LTO, FMA, BMI2 y C++20 para máximo rendimiento.
El binario resultante (grpc-server) es estático y no requiere dependencias en runtime.
---

## 📦 Instalación (GUI de LocalAI)

> 💡 Este backend se instala **manualmente** en la interfaz web de LocalAI. No requiere rebuild del sistema.

### Paso 1: Compilar el backend
```bash
# Clonar o copiar este repo en tu entorno de desarrollo
cd /path/to/your/local-repo/ik_llama-cpp

# Compilar usando el único target soportado
make llama-cpp-grpc

# Verificar que se generó el binario estático
ls -lh grpc-server
# → Debe mostrar algo como: -rwxr-xr-x 1 root root 180M ...
```

> ⚠️ **Importante**: El binario funcional es `grpc-server`. El target `llama-cpp-grpc` es el único recomendado y soportado.

### Paso 2: Copiar al directorio de backends
```bash
# Crear la carpeta del backend (nombre fijo: `cpu-ikllama-cpp`)
sudo mkdir -p /opt/local-path-backends/cpu-ikllama-cpp

# Copiar el binario principal
sudo cp grpc-server /opt/local-path-backends/cpu-ikllama-cpp/

# Copiar las dependencias dinámicas (desde cpu-llama-cpp ya instalado)
sudo cp -r /opt/local-path-backends/cpu-llama-cpp/lib/* /opt/local-path-backends/cpu-ikllama-cpp/lib/

# Verificar
ls /opt/local-path-backends/cpu-ikllama-cpp/
# → Debe mostrar: grpc-server  lib/
```

### Paso 3: Registrar en la GUI de LocalAI
1. Abre la interfaz web de LocalAI (`http://localhost:8080`)
2. Ve a **Backend Management** → **Manual Install**
3. Completa:
   - **OCI Image / URL / Path**: `/opt/local-path-backends/cpu-ikllama-cpp/grpc-server`
   - **Name (required)**: `cpu-ikllama-cpp`
   - **Alias (optional)**: `ikllama`
4. Haz clic en **Install**

✅ El backend quedará disponible para asociarlo a modelos.

---

## 📄 Ejemplo de definición de modelo

Crea un archivo YAML en `/opt/local-path-provisioner/` (o usa la API para cargar modelos):

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
<img width="1382" height="696" alt="image" src="https://github.com/user-attachments/assets/7d21a312-6be9-45f1-b1df-973bdbb62f02" />

---

## ⚠️ Limitaciones conocidas (Alpha)

| Funcionalidad | Estado | Notas |
|---------------|--------|-------|
| **Embeddings** | ⚠️ Parcial | Soporte básico; pooling y normalización en desarrollo activo |
| **Multimodal avanzado** | ⚠️ Limitado | Una imagen/audio por prompt; múltiples medios sin pruebas exhaustivas |
| **GPU** | ❌ No incluido | CPU-only por diseño; requeriría rebuild con `-DGGML_CUDA=ON` |

---

## 🏗️ Detalles técnicos

- **Lenguaje**: C++20 (gRPC, Protobuf, `absl`, `llama.cpp`)
- **Build**: CMake 3.15+ con integración de gRPC estático
- **Protocolo**: gRPC (proto localizado en `LocalAI/backend/backend.proto`)
- **Formatos soportados**: GGUF, GGML, LoRA, MMProj (multimodal)

### Parches críticos aplicados
- `patches/iqk_fa_s_zero.patch`: Evita *crash* en slots vacíos durante paralelismo (esencial para entornos educativos con múltiples usuarios)

---

## 📚 ¿Por qué este backend?

Este proyecto responde a necesidades específicas en entornos educativos:
- ✅ **Privacidad total**: Todo el procesamiento ocurre localmente  
- ✅ **Costo cero**: Sin dependencia de APIs externas  
- ✅ **Escalable**: Optimizado para servidores dedicados (dual-socket Xeon)  
- ✅ **Mantenible**: Build estático, sin dependencias en runtime  

Forma parte del sistema **Cuyex**, desarrollado en el *Colegio Santa Rosa de Lima* para democratizar el acceso a IA en escuelas públicas.

---

## 📜 Licencia

Este proyecto usa la licencia **Apache-2.0**.  
Basado en [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (MIT) y [`LocalAI`](https://github.com/mudler/LocalAI) (Apache-2.0).

---

## 🤝 ¿Quieres contribuir?

¡Ayuda a mejorar este backend! Áreas prioritarias:
- Validación robusta de embeddings (pooling, normalización)  
- Pruebas de multimodal avanzado (múltiples imágenes/audio)  
- Documentación técnica (benchmarking, troubleshooting)  
- Optimizaciones para nuevos modelos (Qwen3, Llama 4, etc.)

---

Estoy listo para ayudarte a hacerlo *listo para producción*.
