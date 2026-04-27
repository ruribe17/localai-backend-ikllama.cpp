// llama.cpp gRPC C++ backend server
//
// Ettore Di Giacinto <mudler@localai.io> and llama.cpp authors
//
// This is a gRPC server for llama.cpp compatible with the LocalAI proto
// Note: this is a re-adaptation of the original llama.cpp example/server.cpp for HTTP (https://github.com/ggerganov/llama.cpp/tree/master/examples/server),
// but modified to work with gRPC
//

#include "server-task.h"
#include "server-queue.h"
#include "server-common.h"
#include "server-context.h"

// LocalAI

#include "backend.pb.h"
#include "backend.grpc.pb.h"
#include "common.h"
#include <getopt.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <regex>
#include <atomic>
#include <mutex>
#include <signal.h>
#include <thread>
#include <unordered_set>


#if defined(_WIN32)
#include <windows.h>
#endif


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
// END LocalAI

// Global variables required by server-common.cpp and server-context.cpp
bool server_verbose = false;
bool server_log_json = false;


/////////////////////////////////
////////////////////////////////
//////// LOCALAI code starts below here
/////////////////////////////////
////////////////////////////////

bool loaded_model; // TODO: add a mutex for this, but happens only once loading the model

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// Forward declarations
static void start_llama_server(server_context& ctx_server);
static json parse_options(bool streaming, const backend::PredictOptions* predict, const gpt_params& params_base, llama_context* ctx);
static ggml_type kv_cache_type_from_str(const std::string & s);
static std::string get_all_kv_cache_types();
static void params_parse(server_context& ctx_server, const backend::ModelOptions* request, gpt_params & params);

// Helper to get n_ctx from slot
static int32_t get_n_ctx_from_slot(const server_context& ctx) {
    if (ctx.slots.empty()) {
        return ctx.n_ctx;
    }
    return ctx.slots[0].n_ctx;
}

static void start_llama_server(server_context& ctx_server) {

    LOG_INF("%s: starting llama server\n", __func__);

    LOG_INF("%s: waiting for model to be loaded\n", __func__);
    // Wait for model to be loaded first
    while (!loaded_model) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    LOG_INF("%s: model loaded\n", __func__);

    shutdown_handler = [&](int) {
        // this will unblock start_loop()
        ctx_server.queue_tasks.terminate();
    };

    // TODO: refactor in common/console
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // Register task callbacks before starting the loop
    // Without these, start_loop() calls empty std::function and throws bad_function_call
    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
    });
    ctx_server.queue_tasks.on_finish_multitask(std::bind(
        &server_context::on_finish_multitask, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots, &ctx_server));
    ctx_server.queue_results.on_multitask_update(std::bind(
        &server_queue::update_multitask,
        &ctx_server.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));

    // this call blocks the main thread until queue_tasks.terminate() is called

    // Registrar callbacks antes de start_loop() (igual que server.cpp)
    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
    });

    ctx_server.queue_tasks.on_finish_multitask(std::bind(
        &server_context::on_finish_multitask,
        &ctx_server,
        std::placeholders::_1));

    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots,
        &ctx_server));

    ctx_server.queue_results.on_multitask_update(std::bind(
        &server_queue::update_multitask,
        &ctx_server.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));


    // Registrar callbacks antes de start_loop() (igual que server.cpp)
    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
    });

    ctx_server.queue_tasks.on_finish_multitask(std::bind(
        &server_context::on_finish_multitask,
        &ctx_server,
        std::placeholders::_1));

    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots,
        &ctx_server));

    ctx_server.queue_results.on_multitask_update(std::bind(
        &server_queue::update_multitask,
        &ctx_server.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));

    ctx_server.queue_tasks.start_loop();
}

json parse_options(bool streaming, const backend::PredictOptions* predict, const gpt_params& params_base, llama_context* ctx)
{

    // Create now a json data from the prediction options instead
    //
    json data;
    data["stream"] = streaming;
    data["cache_prompt"] = predict->promptcacheall();
    data["n_predict"] = predict->tokens() == 0 ? -1 : predict->tokens();
    data["top_k"] = predict->topk();
    data["top_p"] = predict->topp();
    data["typical_p"] = predict->typicalp();
    data["temperature"] = predict->temperature();
    data["repeat_last_n"] = predict->repeat();
    data["repeat_penalty"] = predict->penalty();
    data["frequency_penalty"] = predict->frequencypenalty();
    data["presence_penalty"] = predict->presencepenalty();
    data["mirostat"] = predict->mirostat();
    data["mirostat_tau"] = predict->mirostattau();
    data["mirostat_eta"] = predict->mirostateta();
    data["n_keep"] = predict->nkeep();
    data["seed"] = predict->seed();


    std::string grammar_str = predict->grammar();



    if (!grammar_str.empty()) {
        data["grammar"] = grammar_str;
        SRV_INF("Using grammar: %s\n", grammar_str.c_str());
    }

    // Only set prompt if UseTokenizerTemplate is false or if no Messages are provided
    // When UseTokenizerTemplate is true and Messages are provided, prompt will be set via chat templates in Predict/PredictStream
    if (!predict->usetokenizertemplate() || predict->messages_size() == 0) {
        data["prompt"] = predict->prompt();
    }

    // Extract tools and tool_choice from proto and add to data JSON
    SRV_INF("[TOOLS DEBUG] parse_options: Checking for tools in proto, tools().empty()=%d, tools().size()=%zu\n",
            predict->tools().empty() ? 1 : 0, predict->tools().size());
    if (!predict->tools().empty()) {
        SRV_INF("[TOOLS DEBUG] parse_options: Tools string from proto (first 500 chars): %s\n",
                predict->tools().substr(0, std::min<size_t>(500, predict->tools().size())).c_str());
        try {
            // Parse tools JSON string and add to data
            json tools_json = json::parse(predict->tools());
            data["tools"] = tools_json;
            SRV_INF("Extracted tools from proto: %s\n", predict->tools().c_str());
            // Debug: Log tools count and names
            if (tools_json.is_array()) {
                SRV_INF("[TOOLS DEBUG] parse_options: Successfully parsed %zu tools from Go layer\n", tools_json.size());
                for (size_t i = 0; i < tools_json.size(); i++) {
                    if (tools_json[i].contains("function") && tools_json[i]["function"].contains("name")) {
                        SRV_INF("[TOOLS DEBUG] parse_options: Tool %zu: %s\n", i, tools_json[i]["function"]["name"].get<std::string>().c_str());
                    } else if (tools_json[i].contains("name")) {
                        SRV_INF("[TOOLS DEBUG] parse_options: Tool %zu: %s\n", i, tools_json[i]["name"].get<std::string>().c_str());
                    }
                }
            } else {
                SRV_WRN("[TOOLS DEBUG] parse_options: Parsed tools JSON is not an array: %s\n", tools_json.dump().c_str());
            }
        } catch (const json::parse_error& e) {
            SRV_WRN("Failed to parse tools JSON from proto: %s\n", e.what());
            SRV_WRN("[TOOLS DEBUG] parse_options: Tools string that failed to parse: %s\n", predict->tools().c_str());
        }
    } else {
        SRV_INF("%s", "[TOOLS DEBUG] parse_options: No tools received from Go layer (predict->tools() is empty)\n");
    }

    // Debug: Verify tools are in data after extraction
    if (data.contains("tools")) {
        SRV_INF("[TOOLS DEBUG] parse_options: Tools successfully added to data, count: %zu\n",
                data["tools"].is_array() ? data["tools"].size() : 0);
    } else {
        SRV_INF("%s", "[TOOLS DEBUG] parse_options: WARNING - Tools NOT in data after extraction!\n");
    }
    if (!predict->toolchoice().empty()) {
        try {
            // Parse tool_choice JSON string
            json tool_choice_json = json::parse(predict->toolchoice());
            // tool_choice can be a string ("auto", "none", "required") or an object
            // Store it as-is (string or object) so we can convert object to "required" later when adding to body_json
            if (tool_choice_json.is_string()) {
                data["tool_choice"] = tool_choice_json.get<std::string>();
                SRV_DBG("[TOOLS DEBUG] Received tool_choice from Go layer: %s\n", tool_choice_json.get<std::string>().c_str());
            } else {
                // Store object as-is so we can detect it later and convert to "required"
                data["tool_choice"] = tool_choice_json;
                SRV_DBG("[TOOLS DEBUG] Received tool_choice object from Go layer: %s\n", tool_choice_json.dump().c_str());
            }
            SRV_INF("Extracted tool_choice from proto: %s\n", predict->toolchoice().c_str());
        } catch (const json::parse_error& e) {
            // If parsing fails, treat as string
            data["tool_choice"] = predict->toolchoice();
            SRV_INF("Extracted tool_choice as string: %s\n", predict->toolchoice().c_str());
        }
    }

    // Extract logprobs and top_logprobs from proto and add to JSON data
    // Following server.cpp pattern: logprobs maps to n_probs when provided
    if (predict->logprobs() > 0) {
        data["logprobs"] = predict->logprobs();
        // Map logprobs to n_probs (following server.cpp line 369 pattern)
        // n_probs will be set by params_from_json_cmpl if logprobs is provided
        data["n_probs"] = predict->logprobs();
        SRV_INF("Using logprobs: %d\n", predict->logprobs());
    }
    if (predict->toplogprobs() > 0) {
        data["top_logprobs"] = predict->toplogprobs();
        SRV_INF("Using top_logprobs: %d\n", predict->toplogprobs());
    }

    // Extract logit_bias from proto and add to JSON data
    if (!predict->logitbias().empty()) {
        try {
            // Parse logit_bias JSON string from proto
            json logit_bias_json = json::parse(predict->logitbias());
            // Add to data - llama.cpp server expects it as an object (map)
            data["logit_bias"] = logit_bias_json;
            SRV_INF("Using logit_bias: %s\n", predict->logitbias().c_str());
        } catch (const json::parse_error& e) {
            SRV_ERR("Failed to parse logit_bias JSON from proto: %s\n", e.what());
        }
    }

    data["ignore_eos"] = predict->ignoreeos();
    data["embeddings"] = predict->embeddings();

    // Add the correlationid to json data
    data["correlation_id"] = predict->correlationid();

    // for each image in the request, add the image data
    //
    for (int i = 0; i < predict->images_size(); i++) {
        data["image_data"].push_back(json
            {
                {"id", i},
                {"data",    predict->images(i)},
            });
    }

    // for each audio in the request, add the audio data
    for (int i = 0; i < predict->audios_size(); i++) {
        data["audio_data"].push_back(json
            {
                {"id", i},
                {"data",    predict->audios(i)},
            });
    }

    {
        json stop_array = json::array();
        for (int i = 0; i < predict->stopprompts_size(); ++i) {
            stop_array.push_back(predict->stopprompts(i));
        }
        data["stop"] = stop_array;
        if (!stop_array.empty()) {
            SRV_INF("✅ Stop prompts loaded: %s\n", stop_array.dump().c_str());
        }
    }
    // data["n_probs"] = predict->nprobs();
    //TODO: images,

    // Serialize grammar triggers from server context to JSON array
    if (!params_base.sparams.grammar_triggers.empty()) {
        json grammar_triggers = json::array();
        for (const auto& trigger : params_base.sparams.grammar_triggers) {
            json trigger_json;
            trigger_json["value"] = trigger.value;
            // Always serialize as WORD type since upstream converts WORD to TOKEN internally
            trigger_json["type"] = static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_WORD);
            grammar_triggers.push_back(trigger_json);
        }
        data["grammar_triggers"] = grammar_triggers;
    }

    // Serialize preserved tokens from server context to JSON array
    if (!params_base.sparams.preserved_tokens.empty()) {
        json preserved_tokens = json::array();
        for (const auto& token : params_base.sparams.preserved_tokens) {
            preserved_tokens.push_back(common_token_to_piece(ctx, token));
        }
        data["preserved_tokens"] = preserved_tokens;
    }

    return data;
}


const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::string get_all_kv_cache_types() {
    std::ostringstream msg;
    for (const auto & type : kv_cache_types) {
        msg << ggml_type_name(type) << (&type == &kv_cache_types.back() ? "" : ", ");
    }
    return msg.str();
}

// === params_parse: versión CORREGIDA para ik_llama.cpp con pooling, attention y embd_normalize ===
// Regla de oro: Proto → carga/modelo (fijo), Options → inferencia/optimización (ejecución)
// NO HAY SUPERPOSICIÓN: cada campo proviene solo de request o solo de options

static void params_parse(server_context& /*ctx_server*/, const backend::ModelOptions* request,
                                gpt_params & params) {

    // === 1. CAMPOS SOLO DE REQUEST (PROTO) ===
    // Estos NO deben ser sobrescritos por options
    params.model = request->modelfile();
    if (!request->mmproj().empty()) {
        params.mmproj.path = request->mmproj();
    }
    params.model_alias = request->modelfile();

    // ✅ AQUÍ: Soporte explícito para request->numa()
    if (request->numa()) {
        params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;  // ← CAMBIO AQUÍ: DISTRIBUTE en lugar de SPREAD
        LOG_INF("✅ numa enabled with STRATEGY_DISTRIBUTE (from proto ModelOptions.NUMA)\n");
    } else {
        params.numa = GGML_NUMA_STRATEGY_DISABLED;
        LOG_INF("✅ numa disabled (default)\n");
    }

    if (!request->cachetypekey().empty()) {
        params.cache_type_k = request->cachetypekey();
    }
    if (!request->cachetypevalue().empty()) {
        params.cache_type_v = request->cachetypevalue();
    }

    params.n_ctx = request->contextsize();
    params.n_threads = request->threads();
    params.n_gpu_layers = request->ngpulayers();
    params.n_batch = request->nbatch();

    // ✅ n_ubatch NO VIENE DE PROTO (solo de options), así que inicializamos a -1
    params.n_ubatch = -1;

    // === 2. DEFAULTS DE SERVER OPTIONS (solo valores iniciales, pueden ser sobrescritos por options) ===
    params.ctx_shift = false;
    params.cache_ram_mib = -1;
    params.n_parallel = 1;
    params.graph_reuse = true;
    params.slot_prompt_similarity = 0.1f;
    params.cont_batching = true;
    params.check_tensors = false;
    params.warmup = true;
    params.ctx_checkpoints_n = 8;
   
    LOG_INF("DEBUG: embeddings=%d, options_size=%d\n", request->embeddings(), request->options_size());


    // === 3. LECTURA DE OPTIONS (solo para inferencia/optimización) ===
    for (int i = 0; i < request->options_size(); i++) {
        std::string opt = request->options(i);
        if (opt.empty()) continue;

        size_t colon_pos = opt.find(':');
        std::string optname = opt.substr(0, colon_pos);
        std::string optval_str = (colon_pos == std::string::npos) ? "true" : opt.substr(colon_pos + 1);

        auto is_true = [&optval_str]() {
            return optval_str == "true" || optval_str == "1" || optval_str == "yes" ||
                   optval_str == "on" || optval_str == "enabled";
        };
        auto is_false = [&optval_str]() {
            return optval_str == "false" || optval_str == "0" || optval_str == "no" ||
                   optval_str == "off" || optval_str == "disabled";
        };

        // ✅ IK_LLAMA.CPP CRÍTICOS (solo options)
        if (optname == "n_ubatch") {
            try {
                int val = std::stoi(optval_str);
                if (val >= -1) {
                    params.n_ubatch = val;
                    LOG_INF("✅ n_ubatch set to %d (from options)\n", val);
                } else {
                    LOG_WRN("⚠️ Invalid n_ubatch=%d (must be >= -1), ignoring\n", val);
                }
            } catch (...) {
                LOG_WRN("⚠️ Failed to parse n_ubatch='%s', using default (-1)\n", optval_str.c_str());
            }
            continue;
        }

        if (optname == "attn_max_batch") {
            try {
                int val = std::stoi(optval_str);
                if (val >= 0) {
                    params.attn_max_batch = val;
                    LOG_INF("✅ attn_max_batch set to %d (from options)\n", val);
                } else {
                    LOG_WRN("⚠️ Invalid attn_max_batch=%d (must be >=0), ignoring\n", val);
                }
            } catch (...) {
                LOG_WRN("⚠️ Failed to parse attn_max_batch='%s', using default\n", optval_str.c_str());
            }
            continue;
        }

        if (optname == "grouped_expert_routing") {
            if (is_true()) {
                params.grouped_expert_routing = true;
                LOG_INF("✅ Enabled grouped expert routing (grouped_expert_routing=true)\n");
            } else if (is_false()) {
                params.grouped_expert_routing = false;
                LOG_INF("✅ Disabled grouped expert routing (grouped_expert_routing=false)\n");
            } else {
                LOG_WRN("⚠️ Unknown value for grouped_expert_routing='%s', ignoring\n", optval_str.c_str());
            }
            continue;
        }

        if (optname == "fused_moe_up_gate") {
            if (is_true()) params.fused_moe_up_gate = true;
            else if (is_false()) params.fused_moe_up_gate = false;
            else LOG_WRN("⚠️ Unknown fused_moe_up_gate value='%s', ignoring\n", optval_str.c_str());
            continue;
        }

        if (optname == "fused_up_gate") {
            if (is_true()) params.fused_up_gate = true;
            else if (is_false()) params.fused_up_gate = false;
            else LOG_WRN("⚠️ Unknown fused_up_gate value='%s', ignoring\n", optval_str.c_str());
            continue;
        }

        // === NUEVO: SOLO APLICAR SI EMBEDDINGS = TRUE ===
        params.embedding = request->embeddings();
        if (params.embedding) {
            // attention:non-causal / causal
            if (optname == "attention") {
                if (optval_str == "causal") {
                    params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
                    LOG_INF("✅ attention = causal (from options, embeddings=true)\n");
                } else if (optval_str == "non-causal") {
                    params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
                    LOG_INF("✅ attention = non-causal (from options, embeddings=true)\n");
                } else {
                    LOG_WRN("⚠️ Unknown attention mode '%s' (expected 'causal' or 'non-causal'), ignoring\n", optval_str.c_str());
                }
                continue;
            }

            // pooling:mean / cls / last
            else if (optname == "pooling") {
                if (optval_str == "mean") {
                    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                    LOG_INF("✅ pooling = mean (from options, embeddings=true)\n");
                } else if (optval_str == "cls") {
                    params.pooling_type = LLAMA_POOLING_TYPE_CLS;
                    LOG_INF("✅ pooling = cls (from options, embeddings=true)\n");
                } else if (optval_str == "last") {
                    params.pooling_type = LLAMA_POOLING_TYPE_LAST;
                    LOG_INF("✅ pooling = last (from options, embeddings=true)\n");
                } else {
                    LOG_WRN("⚠️ Unknown pooling mode '%s' (expected 'mean', 'cls', or 'last'), ignoring\n", optval_str.c_str());
                }
                continue;
            }

            // embd_normalize:2 / 1 / 0 / -1
            else if (optname == "embd_normalize") {
                try {
                    int val = std::stoi(optval_str);
                    if (val >= -1) {
                        params.embd_normalize = val;
                        LOG_INF("✅ embd_normalize = %d (from options, embeddings=true)\n", val);
                    } else {
                        LOG_WRN("⚠️ Invalid embd_normalize=%d (must be >= -1), ignoring\n", val);
                    }
                } catch (...) {
                    LOG_WRN("⚠️ Failed to parse embd_normalize='%s', using default (2)\n", optval_str.c_str());
                }
                continue;
            }
        }

        // ✅ OTRAS Opciones (solo options, no proto)
        if (optname == "cache_ram") {
            try {
                params.cache_ram_mib = std::stoi(optval_str);
                LOG_INF("✅ cache_ram set to %d MiB (from options)\n", params.cache_ram_mib);
            } catch (...) {
                LOG_WRN("⚠️ Failed to parse cache_ram='%s', using default (-1)\n", optval_str.c_str());
            }
        }
        else if (optname == "parallel" || optname == "n_parallel") {
            try {
                int val = std::stoi(optval_str);
                params.n_parallel = val;
                if (params.n_parallel > 1) {
                    params.cont_batching = true;
                    LOG_INF("✅ n_parallel=%d → enabled cont_batching\n", val);
                } else {
                    LOG_INF("✅ n_parallel=%d → disabled cont_batching\n", val);
                }
            } catch (...) {
                LOG_WRN("⚠️ Failed to parse n_parallel='%s', using default (1)\n", optval_str.c_str());
            }
        }
        else if (optname == "grpc_servers" || optname == "rpc_servers") {
            params.rpc_servers = optval_str;
            LOG_INF("✅ rpc_servers set to '%s' (from options)\n", optval_str.c_str());
        }
        else if (optname == "context_shift") {
            if (is_true()) params.ctx_shift = true;
            else if (is_false()) params.ctx_shift = false;
            else LOG_WRN("⚠️ Unknown context_shift value='%s', ignoring\n", optval_str.c_str());
        }
        else if (optname == "use_jinja" || optname == "jinja") {
            if (is_true()) params.use_jinja = true;
            else if (is_false()) params.use_jinja = false;
            else LOG_WRN("⚠️ Unknown jinja value='%s', ignoring\n", optval_str.c_str());
        }
        else if (optname == "slot_prompt_similarity" || optname == "sps") {
            try {
                params.slot_prompt_similarity = std::stof(optval_str);
                LOG_INF("✅ slot_prompt_similarity set to %.3f (from options)\n", params.slot_prompt_similarity);
            } catch (...) {
                LOG_WRN("⚠️ Failed to parse slot_prompt_similarity='%s'\n", optval_str.c_str());
            }
        }
        else if (optname == "cont_batching" || optname == "continuous_batching") {
            if (is_true()) params.cont_batching = true;
            else if (is_false()) params.cont_batching = false;
            else LOG_WRN("⚠️ Unknown cont_batching value='%s', ignoring\n", optval_str.c_str());
        }
        else if (optname == "check_tensors") {
            if (is_true()) params.check_tensors = true;
            else if (is_false()) params.check_tensors = false;
            else LOG_WRN("⚠️ Unknown check_tensors value='%s', ignoring\n", optval_str.c_str());
        }
        else if (optname == "warmup") {
            if (is_true()) params.warmup = true;
            else if (is_false()) params.warmup = false;
            else LOG_WRN("⚠️ Unknown warmup value='%s', ignoring\n", optval_str.c_str());
        }
        else if (optname == "n_threads_batch" || optname == "n_threads_batch") {
            try {
                params.n_threads_batch = std::stoi(optval_str);
                LOG_INF("✅ n_threads_batch set to %d (from options)\n", params.n_threads_batch);
            } catch (...) {
                LOG_WRN("⚠️ Failed to parse n_threads_batch='%s'\n", optval_str.c_str());
            }
        }
        else if (optname == "ctx_checkpoints") {
            try {
                params.ctx_checkpoints_n = std::stoi(optval_str);
                LOG_INF("✅ ctx_checkpoints_n set to %d (from options)\n", params.ctx_checkpoints_n);
            } catch (...) {
                LOG_WRN("⚠️ Failed to parse ctx_checkpoints_n='%s'\n", optval_str.c_str());
            }
        }
        else {
            LOG_WRN("⚠️ Unknown option '%s' (ignored)\n", optname.c_str());
        }
    }

    // === 4. FALLBACKS A VARIABLES DE ENTORNO (solo para campos de options) ===
    if (params.n_parallel == 1) {
        const char *env_parallel = std::getenv("LLAMACPP_PARALLEL");
        if (env_parallel != nullptr) {
            try {
                params.n_parallel = std::stoi(env_parallel);
                if (params.n_parallel > 1) {
                    params.cont_batching = true;
                    LOG_INF("✅ n_parallel set from env LLAMACPP_PARALLEL=%d\n", params.n_parallel);
                }
            } catch (...) { /* ignore */ }
        }
    }

    if (params.rpc_servers.empty()) {
        const char *env_rpc = std::getenv("LLAMACPP_GRPC_SERVERS");
        if (env_rpc != nullptr) {
            params.rpc_servers = std::string(env_rpc);
            LOG_INF("✅ rpc_servers set from env LLAMACPP_GRPC_SERVERS='%s'\n", params.rpc_servers.c_str());
        }
    }

    // === 5. KV OVERRIDES ===
    if (request->overrides_size() > 0) {
        for (int i = 0; i < request->overrides_size(); i++) {
            string_parse_kv_override(request->overrides(i).c_str(), params.kv_overrides);
        }
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    // === 6. OTROS CAMPOS SOLO DE REQUEST (PROTO) ===
    if (!request->tensorsplit().empty()) {
        std::string arg_next = request->tensorsplit();
        const std::regex regex{ R"([,/]+)" };
        std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
        std::vector<std::string> split_arg{ it, {} };

        GGML_ASSERT(split_arg.size() <= llama_max_devices());
        for (size_t i_device = 0; i_device < llama_max_devices(); ++i_device) {
            if (i_device < split_arg.size()) {
                try {
                    params.tensor_split[i_device] = std::stof(split_arg[i_device]);
                } catch (...) {
                    params.tensor_split[i_device] = 0.0f;
                }
            } else {
                params.tensor_split[i_device] = 0.0f;
            }
        }
    }

    if (!request->maingpu().empty()) {
        try {
            params.main_gpu = std::stoi(request->maingpu());
        } catch (...) {
            LOG_WRN("⚠️ Invalid main_gpu='%s'\n", request->maingpu().c_str());
        }
    }

    if (!request->loraadapter().empty() && !request->lorabase().empty()) {
        float scale_factor = 1.0f;
        if (request->lorascale() != 0.0f) {
            scale_factor = request->lorascale();
        }
        std::string model_path = params.model;
        std::string model_dir = model_path.substr(0, model_path.find_last_of("/\\"));
        llama_lora_adapter_info lora_info;
        lora_info.path = model_dir + "/" + request->loraadapter();
        lora_info.scale = scale_factor;
        params.lora_adapters.push_back(std::move(lora_info));
        LOG_INF("✅ LoRA adapter loaded: %s (scale=%.2f)\n", lora_info.path.c_str(), lora_info.scale);
    }

    params.use_mlock = request->mlock();
    params.use_mmap = request->mmap();

    if (request->flashattention() == "on" || request->flashattention() == "enabled") {
        params.flash_attn = true;
    } else if (request->flashattention() == "off" || request->flashattention() == "disabled") {
        params.flash_attn = false;
    } else if (request->flashattention() == "auto") {
        params.flash_attn = true;
    }

    params.no_kv_offload = request->nokvoffload();

    params.embedding = request->embeddings() || request->reranking();
    if (request->reranking()) {
        params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    }

    if (request->ropescaling() == "none") {
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    } else if (request->ropescaling() == "yarn") {
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    } else if (request->ropescaling() == "linear") {
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    }

    if (request->yarnextfactor() != 0.0f) params.yarn_ext_factor = request->yarnextfactor();
    if (request->yarnattnfactor() != 0.0f) params.yarn_attn_factor = request->yarnattnfactor();
    if (request->yarnbetafast() != 0.0f) params.yarn_beta_fast = request->yarnbetafast();
    if (request->yarnbetaslow() != 0.0f) params.yarn_beta_slow = request->yarnbetaslow();

    // ✅ rope_freq_base y rope_freq_scale SOLO DE REQUEST (proto)
    // NO se sobrescriben con options (siguiendo regla de oro)
    if (request->ropefreqbase() != 0.0f) {
        params.rope_freq_base = request->ropefreqbase();
    }
    if (request->ropefreqscale() != 0.0f) {
        params.rope_freq_scale = request->ropefreqscale();
    }

    if (request->grammartriggers_size() > 0) {
        for (int i = 0; i < request->grammartriggers_size(); i++) {
            const auto & word = request->grammartriggers(i).word();
            common_grammar_trigger trigger;
            trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
            trigger.value = word;
            params.sparams.grammar_triggers.push_back(std::move(trigger));
        }
    }
}

// ✅ Lógica inteligente para n_ubatch (solo si no se definió en Options)    if (params.n_ubatch == -1) {        if (params.n_batch < 512) {            params.n_ubatch = params.n_batch;        } else {            params.n_ubatch = 512;  // valor por defecto estándar        }    }


// GRPC Server start
class BackendServiceImpl final : public backend::Backend::Service {
private:
    server_context& ctx_server;
    gpt_params params_base; // Store copy of params_base, set after model load

public:
    BackendServiceImpl(server_context& ctx) : ctx_server(ctx) {}

    grpc::Status Health(ServerContext* /*context*/, const backend::HealthMessage* /*request*/, backend::Reply* reply) override {
        // Implement Health RPC
        reply->set_message("OK");
        return Status::OK;
    }

    grpc::Status LoadModel(ServerContext* /*context*/, const backend::ModelOptions* request, backend::Result* result) override {
        // Implement LoadModel RPC
        LOG_INF("🔍 LoadModel received: model='%s', options_size=%d\n",
            request->modelfile().c_str(), request->options_size());

        // Depurar opciones:
        for (int i = 0; i < request->options_size(); i++) {
            SRV_INF("  [option %d] = '%s'\n", i, request->options(i).c_str());
        }

        gpt_params params;
        params_parse(ctx_server, request, params);

        // ✅ ACTIVAR REPACK SIEMPRE (sin depender de options/proto)
        params.repack_tensors = false;

        gpt_params_parse_from_env(params);

        llama_backend_init();
        llama_numa_init(params.numa);


        LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.n_threads, params.n_threads_batch, std::thread::hardware_concurrency());
        LOG_INF("\n");
        LOG_INF("%s\n", gpt_params_get_system_info(params).c_str());
        LOG_INF("\n");

        // load the model
        bool load_success = ctx_server.load_model(params);

        if (!load_success) {
            std::string error_msg = "Failed to load model: " + params.model;
            if (!params.mmproj.path.empty()) {
                error_msg += " (with mmproj: " + params.mmproj.path + ")";
            }
            if (params.speculative.has_dft() && !params.speculative.mparams_dft.path.empty()) {
                error_msg += " (with draft model: " + params.speculative.mparams_dft.path + ")";
            }
            error_msg += ". Model file may not exist or be invalid.";

            result->set_message(error_msg);
            result->set_success(false);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        // === NUEVO: DETECCIÓN AUTOMÁTICA DE TIPO DE ATENCIÓN Y POOLING ===
        // Solo si no se especificó explícitamente y estamos en modo embeddings
        if (params.embedding) {
            // Detectar pooling_type desde metadatos del modelo
            if (params.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
                char buf[16];  // buffer suficiente para "mean", "cls", "last", "none"
                int32_t len = llama_model_meta_val_str(ctx_server.model, "pooling.pooling_type", buf, sizeof(buf));
                std::string pooling_in_model = (len > 0) ? std::string(buf, len) : "none";

                if (pooling_in_model == "mean") {
                    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                    LOG_INF("✅ Auto-detected pooling = mean from model metadata\n");
                } else if (pooling_in_model == "cls") {
                    params.pooling_type = LLAMA_POOLING_TYPE_CLS;
                    LOG_INF("✅ Auto-detected pooling = cls from model metadata\n");
                } else if (pooling_in_model == "last") {
                    params.pooling_type = LLAMA_POOLING_TYPE_LAST;
                    LOG_INF("✅ Auto-detected pooling = last from model metadata\n");
                } else {
                    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;  // ← FORZAR MEAN POR DEFECTO!
                    LOG_INF("⚠️ Model has no pooling metadata, forcing pooling = mean (expected for embeddings)\n");
                }
            }

            // Detectar attention_type desde metadatos del modelo
            if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
                char buf[16];  // buffer suficiente para "causal", "non-causal"
                int32_t len = llama_model_meta_val_str(ctx_server.model, "attention.attention_type", buf, sizeof(buf));
                std::string attn_in_model = (len > 0) ? std::string(buf, len) : "causal";

                if (attn_in_model == "non-causal") {
                    params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
                    LOG_INF("✅ Auto-detected attention = non-causal from model metadata\n");
                } else {
                    params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;  // ← FORZAR NON-CAUSAL POR DEFECTO!
                    LOG_INF("⚠️ Model has causal attention, forcing attention = non-causal (expected for embeddings)\n");
                }
            }
        }

        // Process grammar triggers now that vocab is available
        if (!params.sparams.grammar_triggers.empty()) {
            std::vector<common_grammar_trigger> processed_triggers;
            for (const auto& trigger : params.sparams.grammar_triggers) {
                if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    auto ids = common_tokenize(ctx_server.ctx, trigger.value, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        // Add the token to preserved_tokens if not already present
                        if (params.sparams.preserved_tokens.find(token) == params.sparams.preserved_tokens.end()) {
                            params.sparams.preserved_tokens.insert(token);
                            LOG_INF("Added grammar trigger token to preserved tokens: %d (`%s`)\n", token, trigger.value.c_str());
                        }
                        LOG_INF("Grammar trigger token: %d (`%s`)\n", token, trigger.value.c_str());
                        common_grammar_trigger processed_trigger;
                        processed_trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        processed_trigger.value = trigger.value;
                        processed_trigger.token = token;
                        processed_triggers.push_back(std::move(processed_trigger));
                    } else {
                        LOG_INF("Grammar trigger word: `%s`\n", trigger.value.c_str());
                        processed_triggers.push_back(trigger);
                    }
                } else {
                    processed_triggers.push_back(trigger);
                }
            }
            // Update the grammar triggers in params
            params.sparams.grammar_triggers = std::move(processed_triggers);
        }

        // Sincronizar params_base del server_context
        ctx_server.params_base = params;

        // Inicializar slots y estado interno
        ctx_server.init();
        result->set_message("Loading succeeded");
        result->set_success(true);
        loaded_model = true;
        // Store copy of params_base for use in parse_options and other methods
        params_base = params;

        return Status::OK;
    }

    // Helper function to extract logprobs from JSON response
    static json extract_logprobs_from_json(const json& res_json) {
        json logprobs_json = json::object();

        // Check for OAI-compatible format: choices[0].logprobs
        if (res_json.contains("choices") && res_json["choices"].is_array() &&
            res_json["choices"].size() > 0 && res_json["choices"][0].contains("logprobs")) {
            logprobs_json = res_json["choices"][0]["logprobs"];
        }
        // Check for non-OAI format: completion_probabilities
        else if (res_json.contains("completion_probabilities")) {
            // Convert completion_probabilities to OAI format
            logprobs_json["content"] = res_json["completion_probabilities"];
        }
        // Check for direct logprobs field
        else if (res_json.contains("logprobs")) {
            logprobs_json = res_json["logprobs"];
        }

        return logprobs_json;
    }

    // Helper: build slot_params from json data for a completion task
    static slot_params build_slot_params(const json& data) {
        slot_params p;
        p.stream          = json_value(data, "stream", false);
        p.cache_prompt    = json_value(data, "cache_prompt", true);
        p.n_keep          = json_value(data, "n_keep", 0);
        p.n_predict       = json_value(data, "n_predict", -1);
        if (data.contains("stop") && data["stop"].is_array()) {
            for (const auto& s : data["stop"]) {
                if (s.is_string()) {
                    p.antiprompt.push_back(s.get<std::string>());
                }
            }
        }
        if (data.contains("oaicompat_cmpl_id")) {
            p.oaicompat_cmpl_id = data["oaicompat_cmpl_id"].get<std::string>();
        }
        return p;
    }

    grpc::Status PredictStream(grpc::ServerContext* context, const backend::PredictOptions* request, grpc::ServerWriter<backend::Reply>* writer) override {
        if (params_base.model.empty()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Model not loaded");
        }
        if (params_base.embedding) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Embedding is not supported in streaming mode");
        }

        json data = parse_options(true, request, params_base, ctx_server.ctx);
        auto completion_id = gen_chatcmplid();

        // Declared OUTSIDE try so the streaming loop below can access it
        int first_task_id = -1;

        try {
            std::vector<server_task> tasks;
            std::string prompt_str;
            std::vector<raw_buffer> files;

            // Handle chat templates when UseTokenizerTemplate is enabled and Messages are provided
            if (request->usetokenizertemplate() && request->messages_size() > 0 && ctx_server.chat_templates != nullptr) {
                // Convert proto Messages to JSON format compatible with oaicompat_chat_params_parse
                json body_json;
                json messages_json = json::array();

                // Find the last user message index to attach images/audio to
                int last_user_msg_idx = -1;
                for (int i = request->messages_size() - 1; i >= 0; i--) {
                    if (request->messages(i).role() == "user") {
                        last_user_msg_idx = i;
                        break;
                    }
                }

                for (int i = 0; i < request->messages_size(); i++) {
                    const auto& msg = request->messages(i);
                    json msg_json;
                    msg_json["role"] = msg.role();

                    bool is_last_user_msg = (i == last_user_msg_idx);
                    bool has_images_or_audio = (request->images_size() > 0 || request->audios_size() > 0);

                    // Handle content - can be string, null, or array
                    // For multimodal content, we'll embed images/audio from separate fields
                    if (!msg.content().empty()) {
                        // Try to parse content as JSON to see if it's already an array
                        json content_val;
                        try {
                            content_val = json::parse(msg.content());
                            // Handle null values - convert to empty string to avoid template errors
                            if (content_val.is_null()) {
                                content_val = "";
                            }
                        } catch (const json::parse_error&) {
                            // Not JSON, treat as plain string
                            content_val = msg.content();
                        }

                        // If content is an object (e.g., from tool call failures), convert to string
                        if (content_val.is_object()) {
                            content_val = content_val.dump();
                        }

                        // If content is a string and this is the last user message with images/audio, combine them
                        if (content_val.is_string() && is_last_user_msg && has_images_or_audio) {
                            json content_array = json::array();
                            // Add text first
                            content_array.push_back({{"type", "text"}, {"text", content_val.get<std::string>()}});
                            // Add images
                            if (request->images_size() > 0) {
                                for (int j = 0; j < request->images_size(); j++) {
                                    json image_chunk;
                                    image_chunk["type"] = "image_url";
                                    json image_url;
                                    image_url["url"] = "data:image/jpeg;base64," + request->images(j);
                                    image_chunk["image_url"] = image_url;
                                    content_array.push_back(image_chunk);
                                }
                            }
                            // Add audios
                            if (request->audios_size() > 0) {
                                for (int j = 0; j < request->audios_size(); j++) {
                                    json audio_chunk;
                                    audio_chunk["type"] = "input_audio";
                                    json input_audio;
                                    input_audio["data"] = request->audios(j);
                                    input_audio["format"] = "wav"; // default, could be made configurable
                                    audio_chunk["input_audio"] = input_audio;
                                    content_array.push_back(audio_chunk);
                                }
                            }
                            msg_json["content"] = content_array;
                        } else {
                            // Use content as-is (already array or not last user message)
                            // Ensure null values are converted to empty string
                            if (content_val.is_null()) {
                                msg_json["content"] = "";
                            } else {
                                msg_json["content"] = content_val;
                            }
                        }
                    } else if (is_last_user_msg && has_images_or_audio) {
                        // If no content but this is the last user message with images/audio, create content array
                        json content_array = json::array();
                        if (request->images_size() > 0) {
                            for (int j = 0; j < request->images_size(); j++) {
                                json image_chunk;
                                image_chunk["type"] = "image_url";
                                json image_url;
                                image_url["url"] = "data:image/jpeg;base64," + request->images(j);
                                image_chunk["image_url"] = image_url;
                                content_array.push_back(image_chunk);
                            }
                        }
                        if (request->audios_size() > 0) {
                            for (int j = 0; j < request->audios_size(); j++) {
                                json audio_chunk;
                                audio_chunk["type"] = "input_audio";
                                json input_audio;
                                input_audio["data"] = request->audios(j);
                                input_audio["format"] = "wav"; // default, could be made configurable
                                audio_chunk["input_audio"] = input_audio;
                                content_array.push_back(audio_chunk);
                            }
                        }
                        msg_json["content"] = content_array;
                    } else if (msg.role() == "tool") {
                        // Tool role messages must have content field set, even if empty
                        // Jinja templates expect content to be a string, not null or object
                        if (msg.content().empty()) {
                            msg_json["content"] = "";
                        } else {
                            json content_val;
                            try {
                                content_val = json::parse(msg.content());
                                if (content_val.is_null()) {
                                    msg_json["content"] = "";
                                } else if (content_val.is_object()) {
                                    msg_json["content"] = content_val.dump();
                                } else if (content_val.is_string()) {
                                    msg_json["content"] = content_val.get<std::string>();
                                } else {
                                    msg_json["content"] = content_val.dump();
                                }
                            } catch (const json::parse_error&) {
                                msg_json["content"] = msg.content();
                            }
                        }
                    } else {
                        // Ensure all messages have content set (fallback for any unhandled cases)
                        // Jinja templates expect content to be present, default to empty string if not set
                        if (!msg_json.contains("content")) {
                            msg_json["content"] = "";
                        }
                    }

                    // Add optional fields for OpenAI-compatible message format
                    if (!msg.name().empty()) {
                        msg_json["name"] = msg.name();
                    }
                    if (!msg.tool_call_id().empty()) {
                        msg_json["tool_call_id"] = msg.tool_call_id();
                    }
                    if (!msg.reasoning_content().empty()) {
                        msg_json["reasoning_content"] = msg.reasoning_content();
                    }
                    if (!msg.tool_calls().empty()) {
                        // Parse tool_calls JSON string and add to message
                        try {
                            json tool_calls = json::parse(msg.tool_calls());
                            msg_json["tool_calls"] = tool_calls;
                            // IMPORTANT: If message has tool_calls but content is empty or not set,
                            // set content to space " " instead of empty string ""
                            if (!msg_json.contains("content") || (msg_json.contains("content") && msg_json["content"].is_string() && msg_json["content"].get<std::string>().empty())) {
                                msg_json["content"] = " ";
                            }
                        } catch (const json::parse_error& e) {
                            // Not JSON, treat as plain string
                        }
                    }

                    // Final safety check: Ensure no message has null content (Jinja templates require strings)
                    if (!msg_json.contains("content") || msg_json["content"].is_null()) {
                        msg_json["content"] = "";
                    } else if (msg.role() == "tool" && msg_json["content"].is_array()) {
                        msg_json["content"] = msg_json["content"].dump();
                    }

                    messages_json.push_back(msg_json);
                }

                // Final safety check: Ensure no message has null content (Jinja templates require strings)
                for (size_t idx = 0; idx < messages_json.size(); idx++) {
                    auto& msg = messages_json[idx];
                    if (msg.contains("content") && msg["content"].is_null()) {
                        msg["content"] = "";
                    } else if (!msg.contains("content")) {
                        msg["content"] = "";
                    } else if (msg["role"] == "tool" && msg["content"].is_array()) {
                        msg["content"] = msg["content"].dump();
                    }
                }

                body_json["messages"] = messages_json;
                body_json["stream"] = true; // PredictStream is always streaming

                // Check if grammar is provided from Go layer (NoGrammar=false)
                bool has_grammar_from_go = data.contains("grammar") &&
                    data["grammar"].is_string() &&
                    !data["grammar"].get<std::string>().empty();

                if (!has_grammar_from_go) {
                    // NoGrammar=true: pass tools and let template generate grammar
                    if (data.contains("tools")) {
                        body_json["tools"] = data["tools"];
                    }
                    if (data.contains("tool_choice")) {
                        if (data["tool_choice"].is_object()) {
                            body_json["tool_choice"] = "required";
                        } else {
                            body_json["tool_choice"] = data["tool_choice"];
                        }
                    } else {
                        body_json["tool_choice"] = "auto";
                    }
                } else {
                    // Grammar is provided from Go layer (NoGrammar=false)
                    body_json["grammar"] = data["grammar"];
                }

                if (data.contains("json_schema")) {
                    body_json["json_schema"] = data["json_schema"];
                }
                if (data.contains("response_format")) {
                    body_json["response_format"] = data["response_format"];
                }
                if (data.contains("chat_template_kwargs")) {
                    body_json["chat_template_kwargs"] = data["chat_template_kwargs"];
                }
                if (data.contains("parallel_tool_calls")) {
                    body_json["parallel_tool_calls"] = data["parallel_tool_calls"];
                }
                if (data.contains("add_generation_prompt")) {
                    body_json["add_generation_prompt"] = data["add_generation_prompt"];
                }

                // Pass sampling parameters to body_json
                if (data.contains("n_predict")) {
                    body_json["max_tokens"] = data["n_predict"];
                }
                if (data.contains("ignore_eos")) {
                    body_json["ignore_eos"] = data["ignore_eos"];
                }
                if (data.contains("stop")) {
                    body_json["stop"] = data["stop"];
                }
                if (data.contains("temperature")) {
                    body_json["temperature"] = data["temperature"];
                }
                if (data.contains("top_p")) {
                    body_json["top_p"] = data["top_p"];
                }
                if (data.contains("frequency_penalty")) {
                    body_json["frequency_penalty"] = data["frequency_penalty"];
                }
                if (data.contains("presence_penalty")) {
                    body_json["presence_penalty"] = data["presence_penalty"];
                }
                if (data.contains("seed")) {
                    body_json["seed"] = data["seed"];
                }
                if (data.contains("logit_bias")) {
                    body_json["logit_bias"] = data["logit_bias"];
                }
                if (data.contains("top_k")) {
                    body_json["top_k"] = data["top_k"];
                }
                if (data.contains("min_p")) {
                    body_json["min_p"] = data["min_p"];
                }

                // Build oaicompat_parser_options for this fork's API
                oaicompat_parser_options oai_opt;
                oai_opt.use_jinja = params_base.use_jinja;
                oai_opt.prefill_assistant = params_base.prefill_assistant;
                oai_opt.reasoning_format = params_base.reasoning_format;
                oai_opt.chat_template_kwargs = params_base.default_template_kwargs;
                oai_opt.tmpls = ctx_server.chat_templates.get();
                oai_opt.allow_image = (ctx_server.mctx != nullptr);
                oai_opt.allow_audio = (ctx_server.mctx != nullptr);

                json parsed_data = oaicompat_chat_params_parse(ctx_server.model, body_json, oai_opt, files);

                // Extract the prompt from parsed data
                prompt_str = parsed_data.at("prompt").get<std::string>();

                // Preserve grammar from Go layer if it was provided (NoGrammar=false)
                json preserved_grammar;
                if (has_grammar_from_go && data.contains("grammar")) {
                    preserved_grammar = data["grammar"];
                }

                // Merge all fields from parsed_data into data
                for (const auto& item : parsed_data.items()) {
                    if (item.key() != "prompt") {
                        if (item.key() == "grammar" && has_grammar_from_go && !preserved_grammar.is_null()) {
                            data["grammar"] = preserved_grammar;
                        } else {
                            data[item.key()] = item.value();
                        }
                    }
                }
            } else {
                // Use prompt directly from data
                if (data.contains("prompt") && data["prompt"].is_string()) {
                    prompt_str = data["prompt"].get<std::string>();
                } else {
                    prompt_str = request->prompt();
                }
            }

            const auto type = SERVER_TASK_TYPE_COMPLETION;

            // If not using chat templates, extract files from image_data/audio_data fields
            if (!request->usetokenizertemplate() || request->messages_size() == 0 || ctx_server.chat_templates == nullptr) {
                const auto &images_data = data.find("image_data");
                if (images_data != data.end() && images_data->is_array())
                {
                    for (const auto &img : *images_data)
                    {
                        auto decoded_data = base64_decode(img["data"].get<std::string>());
                        files.push_back(decoded_data);
                    }
                }

                const auto &audio_data = data.find("audio_data");
                if (audio_data != data.end() && audio_data->is_array())
                {
                    for (const auto &audio : *audio_data)
                    {
                        auto decoded_data = base64_decode(audio["data"].get<std::string>());
                        files.push_back(decoded_data);
                    }
                }
            }

            const bool has_mtmd = ctx_server.mctx != nullptr;

            // process prompt
            std::vector<server_tokens> inputs;
            if (has_mtmd) {
                // multimodal
                inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt_str, files));
            } else {
                 // Everything else, including multimodal completions.
                inputs = tokenize_input_prompts(llama_model_get_vocab(ctx_server.model), ctx_server.mctx, prompt_str, true, true);
            }

            // Build slot_params from data
            slot_params sparams = build_slot_params(data);
            sparams.stream = true;
            sparams.oaicompat_cmpl_id = completion_id;

            // Register waiting task ids
            std::vector<int> task_ids;
            task_ids.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                task_ids.push_back(ctx_server.queue_tasks.get_new_id());
            }
            first_task_id = task_ids[0];
            ctx_server.queue_results.add_waiting_task_id(first_task_id);

            tasks.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                server_task task = server_task(type);

                task.id    = task_ids[i];
                task.index = i;
                task.tokens    = std::move(inputs[i]);
                task.params    = sparams;
                task.id_slot = json_value(data, "id_slot", -1);
                task.data    = data;

                tasks.push_back(std::move(task));
            }

            ctx_server.queue_tasks.post(std::move(tasks));
        } catch (const std::exception & e) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
        }

        if (first_task_id < 0) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "No task was created");
        }

        // ── Streaming loop ──────────────────────────────────────────────────
        bool done = false;
        while (!done) {
            if (context->IsCancelled()) {
                // Ask the server to cancel this task
                {
                    server_task cancel_task(SERVER_TASK_TYPE_CANCEL);
                    cancel_task.id_target = first_task_id;
                    cancel_task.id = ctx_server.queue_tasks.get_new_id();
                    ctx_server.queue_tasks.post(std::move(cancel_task));
                }
                ctx_server.queue_results.remove_waiting_task_id(first_task_id);
                return grpc::Status(grpc::StatusCode::CANCELLED, "Request cancelled by client");
            }

            // Use recv_with_timeout (new ptr system) instead of legacy recv()
            server_task_result_ptr result_ptr = nullptr;
            while (result_ptr == nullptr) {
                if (context->IsCancelled()) {
                    server_task cancel_task(SERVER_TASK_TYPE_CANCEL);
                    cancel_task.id_target = first_task_id;
                    cancel_task.id = ctx_server.queue_tasks.get_new_id();
                    ctx_server.queue_tasks.post(std::move(cancel_task));
                    ctx_server.queue_results.remove_waiting_task_id(first_task_id);
                    return grpc::Status(grpc::StatusCode::CANCELLED, "Request cancelled by client");
                }
                result_ptr = ctx_server.queue_results.recv_with_timeout(
                    std::unordered_set<int>{first_task_id}, 1);
            }

            if (result_ptr->is_error()) {
                ctx_server.queue_results.remove_waiting_task_id(first_task_id);
                json err_json = result_ptr->to_json();
                std::string err = err_json.value("message", "Error during streaming");
                return grpc::Status(grpc::StatusCode::INTERNAL, err);
            }

            // Extract content via cast para obtener texto plano (no Base64)
            json res_json;
            std::string token_text;
            auto* partial = dynamic_cast<server_task_result_cmpl_partial*>(result_ptr.get());
            auto* final_r = dynamic_cast<server_task_result_cmpl_final*>(result_ptr.get());
            if (partial) {
                res_json = partial->to_json_non_oaicompat_partial();
            } else if (final_r) {
                res_json = final_r->to_json_non_oaicompat_final();
            } else {
                res_json = result_ptr->to_json();
            }
            token_text = res_json.value("content", "");

            if (!token_text.empty()) {
                backend::Reply reply;
                reply.set_message(token_text);
                if (!writer->Write(reply)) {
                    // Client disconnected
                    ctx_server.queue_results.remove_waiting_task_id(first_task_id);
                    return grpc::Status(grpc::StatusCode::CANCELLED, "Client disconnected");
                }
            }

            // result_ptr->is_stop() is true on the final result
            if (result_ptr->is_stop()) {
                done = true;

                // Send timing/usage info on the final reply if available
                if (res_json.contains("timings")) {
                    backend::Reply final_reply;
                    final_reply.set_message("");
                    int32_t tokens_predicted = res_json.value("tokens_predicted", 0);
                    int32_t tokens_evaluated = res_json.value("tokens_evaluated", 0);
                    final_reply.set_tokens(tokens_predicted);
                    final_reply.set_prompt_tokens(tokens_evaluated);
                    double timing_prompt = res_json.at("timings").value("prompt_ms", 0.0);
                    double timing_gen    = res_json.at("timings").value("predicted_ms", 0.0);
                    final_reply.set_timing_prompt_processing(timing_prompt);
                    final_reply.set_timing_token_generation(timing_gen);
                    writer->Write(final_reply);
                }
            }
        }

        ctx_server.queue_results.remove_waiting_task_id(first_task_id);
        return grpc::Status::OK;
    }

    grpc::Status Predict(ServerContext* context, const backend::PredictOptions* request, backend::Reply* reply) override {
         if (params_base.model.empty()) {
             return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Model not loaded");
         }
         json data = parse_options(true, request, params_base, ctx_server.ctx);

        data["stream"] = false;
        //Raise error if embeddings is set to true
        if (params_base.embedding) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Embedding is not supported in Predict mode");
        }
        std::cout << "[PREDICT] Received result: " << data.dump(2) << std::endl;
        auto completion_id = gen_chatcmplid();
        int first_task_id = -1;
        try {
            std::vector<server_task> tasks;

            std::string prompt_str;
            std::vector<raw_buffer> files;
            // Handle chat templates when UseTokenizerTemplate is enabled and Messages are provided
            if (request->usetokenizertemplate() && request->messages_size() > 0 && ctx_server.chat_templates != nullptr) {
                // Convert proto Messages to JSON format compatible with oaicompat_chat_params_parse
                json body_json;
                json messages_json = json::array();

                // Find the last user message index to attach images/audio to
                int last_user_msg_idx = -1;
                for (int i = request->messages_size() - 1; i >= 0; i--) {
                    if (request->messages(i).role() == "user") {
                        last_user_msg_idx = i;
                        break;
                    }
                }

                SRV_INF("[CONTENT DEBUG] Predict: Processing %d messages\n", request->messages_size());
                for (int i = 0; i < request->messages_size(); i++) {
                    const auto& msg = request->messages(i);
                    json msg_json;
                    msg_json["role"] = msg.role();

                    SRV_INF("[CONTENT DEBUG] Predict: Message %d: role=%s, content_empty=%d, content_length=%zu\n",
                            i, msg.role().c_str(), msg.content().empty() ? 1 : 0, msg.content().size());
                    if (!msg.content().empty()) {
                        SRV_INF("[CONTENT DEBUG] Predict: Message %d content (first 200 chars): %s\n",
                                i, msg.content().substr(0, std::min<size_t>(200, msg.content().size())).c_str());
                    }

                    bool is_last_user_msg = (i == last_user_msg_idx);
                    bool has_images_or_audio = (request->images_size() > 0 || request->audios_size() > 0);

                    if (!msg.content().empty()) {
                        json content_val;
                        try {
                            content_val = json::parse(msg.content());
                            if (content_val.is_null()) {
                                SRV_INF("[CONTENT DEBUG] Predict: Message %d parsed JSON is null, converting to empty string\n", i);
                                content_val = "";
                            }
                        } catch (const json::parse_error&) {
                            content_val = msg.content();
                        }

                        if (content_val.is_object()) {
                            SRV_INF("[CONTENT DEBUG] Predict: Message %d content is object, converting to string\n", i);
                            content_val = content_val.dump();
                        }

                        if (content_val.is_string() && is_last_user_msg && has_images_or_audio) {
                            json content_array = json::array();
                            content_array.push_back({{"type", "text"}, {"text", content_val.get<std::string>()}});
                            if (request->images_size() > 0) {
                                for (int j = 0; j < request->images_size(); j++) {
                                    json image_chunk;
                                    image_chunk["type"] = "image_url";
                                    json image_url;
                                    image_url["url"] = "data:image/jpeg;base64," + request->images(j);
                                    image_chunk["image_url"] = image_url;
                                    content_array.push_back(image_chunk);
                                }
                            }
                            if (request->audios_size() > 0) {
                                for (int j = 0; j < request->audios_size(); j++) {
                                    json audio_chunk;
                                    audio_chunk["type"] = "input_audio";
                                    json input_audio;
                                    input_audio["data"] = request->audios(j);
                                    input_audio["format"] = "wav";
                                    audio_chunk["input_audio"] = input_audio;
                                    content_array.push_back(audio_chunk);
                                }
                            }
                            msg_json["content"] = content_array;
                        } else {
                            if (content_val.is_null()) {
                                SRV_INF("[CONTENT DEBUG] Predict: Message %d content_val was null, setting to empty string\n", i);
                                msg_json["content"] = "";
                            } else {
                                msg_json["content"] = content_val;
                            }
                        }
                    } else if (is_last_user_msg && has_images_or_audio) {
                        json content_array = json::array();
                        if (request->images_size() > 0) {
                            for (int j = 0; j < request->images_size(); j++) {
                                json image_chunk;
                                image_chunk["type"] = "image_url";
                                json image_url;
                                image_url["url"] = "data:image/jpeg;base64," + request->images(j);
                                image_chunk["image_url"] = image_url;
                                content_array.push_back(image_chunk);
                            }
                        }
                        if (request->audios_size() > 0) {
                            for (int j = 0; j < request->audios_size(); j++) {
                                json audio_chunk;
                                audio_chunk["type"] = "input_audio";
                                json input_audio;
                                input_audio["data"] = request->audios(j);
                                input_audio["format"] = "wav";
                                audio_chunk["input_audio"] = input_audio;
                                content_array.push_back(audio_chunk);
                            }
                        }
                        msg_json["content"] = content_array;
                        SRV_INF("[CONTENT DEBUG] Predict: Message %d created content array with media\n", i);
                    } else if (!msg.tool_calls().empty()) {
                        SRV_INF("[CONTENT DEBUG] Predict: Message %d has tool_calls, setting content to space\n", i);
                        msg_json["content"] = " ";
                    } else if (msg.role() == "tool") {
                        SRV_INF("[CONTENT DEBUG] Predict: Message %d is tool role, content_empty=%d\n", i, msg.content().empty() ? 1 : 0);
                        if (msg.content().empty()) {
                            msg_json["content"] = "";
                        } else {
                            json content_val;
                            try {
                                content_val = json::parse(msg.content());
                                if (content_val.is_null()) {
                                    msg_json["content"] = "";
                                } else if (content_val.is_object()) {
                                    msg_json["content"] = content_val.dump();
                                } else if (content_val.is_string()) {
                                    msg_json["content"] = content_val.get<std::string>();
                                } else {
                                    msg_json["content"] = content_val.dump();
                                }
                            } catch (const json::parse_error&) {
                                msg_json["content"] = msg.content();
                            }
                        }
                    } else {
                        if (!msg_json.contains("content")) {
                            SRV_INF("[CONTENT DEBUG] Predict: Message %d (role=%s): no content field, adding empty string\n",
                                    i, msg.role().c_str());
                            msg_json["content"] = "";
                        }
                    }

                    // Add optional fields for OpenAI-compatible message format
                    if (!msg.name().empty()) {
                        msg_json["name"] = msg.name();
                    }
                    if (!msg.tool_call_id().empty()) {
                        msg_json["tool_call_id"] = msg.tool_call_id();
                    }
                    if (!msg.reasoning_content().empty()) {
                        msg_json["reasoning_content"] = msg.reasoning_content();
                    }
                    if (!msg.tool_calls().empty()) {
                        try {
                            json tool_calls = json::parse(msg.tool_calls());
                            msg_json["tool_calls"] = tool_calls;
                            SRV_INF("[TOOL CALLS DEBUG] Predict: Message %d has tool_calls: %s\n", i, tool_calls.dump().c_str());
                            if (!msg_json.contains("content") || (msg_json.contains("content") && msg_json["content"].is_string() && msg_json["content"].get<std::string>().empty())) {
                                SRV_INF("[CONTENT DEBUG] Predict: Message %d has tool_calls but empty content, setting to space\n", i);
                                msg_json["content"] = " ";
                            }
                            if (tool_calls.is_array()) {
                                for (size_t tc_idx = 0; tc_idx < tool_calls.size(); tc_idx++) {
                                    const auto& tc = tool_calls[tc_idx];
                                    std::string tool_name = "unknown";
                                    std::string tool_args = "{}";
                                    if (tc.contains("function")) {
                                        const auto& func = tc["function"];
                                        if (func.contains("name")) {
                                            tool_name = func["name"].get<std::string>();
                                        }
                                        if (func.contains("arguments")) {
                                            tool_args = func["arguments"].is_string() ?
                                                func["arguments"].get<std::string>() :
                                                func["arguments"].dump();
                                        }
                                    } else if (tc.contains("name")) {
                                        tool_name = tc["name"].get<std::string>();
                                        if (tc.contains("arguments")) {
                                            tool_args = tc["arguments"].is_string() ?
                                                tc["arguments"].get<std::string>() :
                                                tc["arguments"].dump();
                                        }
                                    }
                                    SRV_INF("[TOOL CALLS DEBUG] Predict: Message %d, tool_call %zu: name=%s, arguments=%s\n",
                                            i, tc_idx, tool_name.c_str(), tool_args.c_str());
                                }
                            }
                        } catch (const json::parse_error& e) {
                            SRV_WRN("Failed to parse tool_calls JSON: %s\n", e.what());
                        }
                    }

                    // Debug: Log final content state before adding to array
                    if (msg_json.contains("content")) {
                        if (msg_json["content"].is_null()) {
                            SRV_INF("[CONTENT DEBUG] Predict: Message %d FINAL STATE: content is NULL - THIS WILL CAUSE ERROR!\n", i);
                        } else {
                            SRV_INF("[CONTENT DEBUG] Predict: Message %d FINAL STATE: content type=%s, has_value=%d\n",
                                    i, msg_json["content"].is_string() ? "string" :
                                       msg_json["content"].is_array() ? "array" :
                                       msg_json["content"].is_object() ? "object" : "other",
                                    msg_json["content"].is_null() ? 0 : 1);
                        }
                    } else {
                        SRV_INF("[CONTENT DEBUG] Predict: Message %d FINAL STATE: NO CONTENT FIELD - THIS WILL CAUSE ERROR!\n", i);
                    }

                    messages_json.push_back(msg_json);
                }

                // Final safety check
                SRV_INF("[CONTENT DEBUG] Predict: Running final safety check on %zu messages\n", messages_json.size());
                for (size_t idx = 0; idx < messages_json.size(); idx++) {
                    auto& msg = messages_json[idx];
                    std::string role_str = msg.contains("role") ? msg["role"].get<std::string>() : "unknown";
                    if (msg.contains("content") && msg["content"].is_null()) {
                        SRV_INF("[CONTENT DEBUG] Predict: Safety check found message %zu (role=%s) with NULL content, converting to empty string\n", idx, role_str.c_str());
                        msg["content"] = "";
                    } else if (!msg.contains("content")) {
                        SRV_INF("[CONTENT DEBUG] Predict: Safety check found message %zu (role=%s) without content field, adding empty string\n", idx, role_str.c_str());
                        msg["content"] = "";
                    }
                }

                int tool_msg_count = 0;
                for (const auto& msg : messages_json) {
                    if (msg.contains("role") && msg["role"] == "tool") {
                        tool_msg_count++;
                    }
                }
                SRV_DBG("[TOOLS DEBUG] Predict: Built %d tool messages out of %zu total messages\n", tool_msg_count, messages_json.size());
                SRV_DBG("[CONVERSATION DEBUG] Predict: Full messages array:\n%s\n", messages_json.dump(2).c_str());

                body_json["messages"] = messages_json;
                body_json["stream"] = false;

                bool has_grammar_from_go = data.contains("grammar") &&
                    data["grammar"].is_string() &&
                    !data["grammar"].get<std::string>().empty();

                SRV_INF("[TOOLS DEBUG] Predict: has_grammar_from_go=%d, data.contains(\"tools\")=%d, data.contains(\"grammar\")=%d\n",
                        has_grammar_from_go ? 1 : 0,
                        data.contains("tools") ? 1 : 0,
                        data.contains("grammar") ? 1 : 0);

                if (!has_grammar_from_go) {
                    if (data.contains("tools")) {
                        body_json["tools"] = data["tools"];
                        std::string tools_str = data["tools"].dump();
                        SRV_INF("Using tools from data (NoGrammar=true): %s\n", tools_str.c_str());
                    } else {
                        SRV_WRN("%s", "No tools found in data - tool calls will not work without tools field\n");
                    }
                    if (data.contains("tool_choice")) {
                        if (data["tool_choice"].is_string()) {
                            body_json["tool_choice"] = data["tool_choice"].get<std::string>();
                        } else if (data["tool_choice"].is_object()) {
                            body_json["tool_choice"] = "required";
                            std::string tool_choice_obj_str = data["tool_choice"].dump();
                            SRV_INF("Converted object tool_choice to 'required': %s\n", tool_choice_obj_str.c_str());
                        } else {
                            body_json["tool_choice"] = data["tool_choice"].dump();
                        }
                        std::string tool_choice_str = body_json["tool_choice"].get<std::string>();
                        SRV_INF("Using tool_choice: %s\n", tool_choice_str.c_str());
                    } else {
                        body_json["tool_choice"] = "auto";
                    }
                } else {
                    SRV_INF("%s", "Grammar provided from Go layer - using it instead of template-generated grammar\n");
                }

                if (data.contains("json_schema")) {
                    body_json["json_schema"] = data["json_schema"];
                }
                if (has_grammar_from_go) {
                    body_json["grammar"] = data["grammar"];
                }
                if (data.contains("response_format")) {
                    body_json["response_format"] = data["response_format"];
                }
                if (data.contains("chat_template_kwargs")) {
                    body_json["chat_template_kwargs"] = data["chat_template_kwargs"];
                }
                if (data.contains("parallel_tool_calls")) {
                    body_json["parallel_tool_calls"] = data["parallel_tool_calls"];
                }
                if (data.contains("add_generation_prompt")) {
                    body_json["add_generation_prompt"] = data["add_generation_prompt"];
                }

                if (data.contains("n_predict")) {
                    body_json["max_tokens"] = data["n_predict"];
                }
                if (data.contains("ignore_eos")) {
                    body_json["ignore_eos"] = data["ignore_eos"];
                }
                if (data.contains("stop")) {
                    body_json["stop"] = data["stop"];
                }
                if (data.contains("temperature")) {
                    body_json["temperature"] = data["temperature"];
                }
                if (data.contains("top_p")) {
                    body_json["top_p"] = data["top_p"];
                }
                if (data.contains("frequency_penalty")) {
                    body_json["frequency_penalty"] = data["frequency_penalty"];
                }
                if (data.contains("presence_penalty")) {
                    body_json["presence_penalty"] = data["presence_penalty"];
                }
                if (data.contains("seed")) {
                    body_json["seed"] = data["seed"];
                }
                if (data.contains("logit_bias")) {
                    body_json["logit_bias"] = data["logit_bias"];
                }
                if (data.contains("top_k")) {
                    body_json["top_k"] = data["top_k"];
                }
                if (data.contains("min_p")) {
                    body_json["min_p"] = data["min_p"];
                }

                SRV_DBG("[CONVERSATION DEBUG] Predict: Full body_json before oaicompat_chat_params_parse:\n%s\n", body_json.dump(2).c_str());

                if (body_json.contains("messages") && body_json["messages"].is_array()) {
                    for (size_t idx = 0; idx < body_json["messages"].size(); idx++) {
                        auto& msg = body_json["messages"][idx];
                        std::string role_str = msg.contains("role") ? msg["role"].get<std::string>() : "unknown";
                        if (msg.contains("content")) {
                            if (msg["content"].is_null()) {
                                msg["content"] = "";
                            } else if (role_str == "tool" && msg["content"].is_array()) {
                                msg["content"] = msg["content"].dump();
                            } else if (!msg["content"].is_string() && !msg["content"].is_array()) {
                                if (msg["content"].is_object()) {
                                    msg["content"] = msg["content"].dump();
                                } else {
                                    msg["content"] = "";
                                }
                            }
                        } else {
                            msg["content"] = "";
                        }
                    }
                }

                // Build oaicompat_parser_options for this fork's API
                oaicompat_parser_options oai_opt;
                oai_opt.use_jinja = params_base.use_jinja;
                oai_opt.prefill_assistant = params_base.prefill_assistant;
                oai_opt.reasoning_format = params_base.reasoning_format;
                oai_opt.chat_template_kwargs = params_base.default_template_kwargs;
                oai_opt.tmpls = ctx_server.chat_templates.get();
                oai_opt.allow_image = (ctx_server.mctx != nullptr);
                oai_opt.allow_audio = (ctx_server.mctx != nullptr);

                json parsed_data = oaicompat_chat_params_parse(ctx_server.model, body_json, oai_opt, files);

                if (parsed_data.contains("tools")) {
                    SRV_DBG("[TOOLS DEBUG] Predict: After oaicompat_chat_params_parse - tools count: %zu\n",
                            parsed_data["tools"].is_array() ? parsed_data["tools"].size() : 0);
                } else {
                    SRV_DBG("%s", "[TOOLS DEBUG] Predict: After oaicompat_chat_params_parse - no tools in parsed_data\n");
                }

                prompt_str = parsed_data.at("prompt").get<std::string>();

                json preserved_grammar;
                if (has_grammar_from_go && data.contains("grammar")) {
                    preserved_grammar = data["grammar"];
                }

                for (const auto& item : parsed_data.items()) {
                    if (item.key() != "prompt") {
                        if (item.key() == "grammar" && has_grammar_from_go && !preserved_grammar.is_null()) {
                            data["grammar"] = preserved_grammar;
                        } else {
                            data[item.key()] = item.value();
                        }
                    }
                }

                if (data.contains("parse_tool_calls")) {
                    SRV_DBG("[TOOLS DEBUG] Predict: parse_tool_calls=%s\n", data["parse_tool_calls"].get<bool>() ? "true" : "false");
                }
            } else {
                // Use prompt directly from data
                if (data.contains("prompt") && data["prompt"].is_string()) {
                    prompt_str = data["prompt"].get<std::string>();
                } else {
                    prompt_str = request->prompt();
                }
            }

            const auto type = SERVER_TASK_TYPE_COMPLETION;

            // If not using chat templates, extract files from image_data/audio_data fields
            if (!request->usetokenizertemplate() || request->messages_size() == 0 || ctx_server.chat_templates == nullptr) {
                const auto &images_data = data.find("image_data");
                if (images_data != data.end() && images_data->is_array())
                {
                    std::cout << "[PREDICT] Processing " << images_data->size() << " images" << std::endl;
                    for (const auto &img : *images_data)
                    {
                        std::cout << "[PREDICT] Processing image" << std::endl;
                        auto decoded_data = base64_decode(img["data"].get<std::string>());
                        files.push_back(decoded_data);
                    }
                }

                const auto &audio_data = data.find("audio_data");
                if (audio_data != data.end() && audio_data->is_array())
                {
                    for (const auto &audio : *audio_data)
                    {
                        auto decoded_data = base64_decode(audio["data"].get<std::string>());
                        files.push_back(decoded_data);
                    }
                }
            }

            const bool has_mtmd = ctx_server.mctx != nullptr;

            // process prompt
            std::vector<server_tokens> inputs;
            if (has_mtmd) {
                inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt_str, files));
            } else {
                inputs = tokenize_input_prompts(llama_model_get_vocab(ctx_server.model), ctx_server.mctx, prompt_str, true, true);
            }

            slot_params sparams = build_slot_params(data);
            sparams.stream = false;
            sparams.oaicompat_cmpl_id = completion_id;

            // Allocate task IDs and register first one for results
            std::vector<int> task_ids;
            task_ids.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                task_ids.push_back(ctx_server.queue_tasks.get_new_id());
            }
            first_task_id = task_ids[0];
            ctx_server.queue_results.add_waiting_task_id(first_task_id);

            tasks.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                server_task task = server_task(type);

                task.id    = task_ids[i];
                task.index = i;
                task.tokens    = std::move(inputs[i]);
                task.params    = sparams;
                task.id_slot = json_value(data, "id_slot", -1);
                task.data    = data;

                tasks.push_back(std::move(task));
            }

            ctx_server.queue_tasks.post(std::move(tasks));
        } catch (const std::exception & e) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
        }


        std::cout << "[DEBUG] Waiting for results..." << std::endl;

        // Wait for final result
        if (first_task_id < 0) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "No task was created");
        }

        // Use recv_with_timeout (new ptr system) instead of legacy recv()
        server_task_result_ptr result_ptr = nullptr;
        while (result_ptr == nullptr) {
            if (context->IsCancelled()) {
                ctx_server.queue_results.remove_waiting_task_id(first_task_id);
                return grpc::Status(grpc::StatusCode::CANCELLED, "Request cancelled by client");
            }
            result_ptr = ctx_server.queue_results.recv_with_timeout(
                std::unordered_set<int>{first_task_id}, 1);
        }
        ctx_server.queue_results.remove_waiting_task_id(first_task_id);

        if (result_ptr->is_error()) {
            json err_json = result_ptr->to_json();
            std::string err_msg = err_json.value("message", "Error occurred");
            std::cout << "[DEBUG] Error in results: " << err_msg << std::endl;
            reply->set_message(err_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, err_msg);
        }

        std::cout << "[DEBUG] Received result" << std::endl;

        // Cast al tipo concreto para obtener content como texto plano (no Base64)
        std::string content_str;
        auto* final_result = dynamic_cast<server_task_result_cmpl_final*>(result_ptr.get());
        if (final_result) {
            json result_json = final_result->to_json_non_oaicompat_final();
            content_str = result_json.value("content", "");
        } else {
            json result_json = result_ptr->to_json();
            content_str = result_json.value("content", "");
        }
        reply->set_message(content_str);

        // Re-obtener result_json para timings/logprobs
        json result_json = result_ptr->to_json();

        int32_t tokens_predicted = result_json.value("tokens_predicted", 0);
        reply->set_tokens(tokens_predicted);
        int32_t tokens_evaluated = result_json.value("tokens_evaluated", 0);
        reply->set_prompt_tokens(tokens_evaluated);

        if (result_json.contains("timings")) {
            double timing_prompt_processing = result_json.at("timings").value("prompt_ms", 0.0);
            reply->set_timing_prompt_processing(timing_prompt_processing);
            double timing_token_generation = result_json.at("timings").value("predicted_ms", 0.0);
            reply->set_timing_token_generation(timing_token_generation);
        }

        // Extract and set logprobs if present
        json logprobs_json = extract_logprobs_from_json(result_json);
        if (!logprobs_json.empty() && !logprobs_json.is_null()) {
            std::string logprobs_str = logprobs_json.dump();
            reply->set_logprobs(logprobs_str);
        }

        std::cout << "[DEBUG] Predict request completed successfully" << std::endl;

        return grpc::Status::OK;
    }

    grpc::Status Embedding(ServerContext* context, const backend::PredictOptions* request, backend::EmbeddingResult* embeddingResult) override {
        if (params_base.model.empty()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Model not loaded");
        }
        json body = parse_options(false, request, params_base, ctx_server.ctx);

        body["stream"] = false;

        // for the shape of input/content, see tokenize_input_prompts()
        json prompt = body.at("embeddings");

        auto tokenized_prompts = tokenize_input_prompts(llama_model_get_vocab(ctx_server.model), ctx_server.mctx, prompt, true, true);
        for (const auto & tokens : tokenized_prompts) {
            if (tokens.empty()) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Input content cannot be empty");
            }
        }

        // ✅ USAR params_base.embd_normalize EN LUGAR DE HARDCODEADO
        int embd_normalize = (params_base.embd_normalize > 0) ? params_base.embd_normalize : 2;
        // create and queue the task

        std::vector<int> task_ids;
        task_ids.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            task_ids.push_back(ctx_server.queue_tasks.get_new_id());
        }

        {
            std::vector<server_task> tasks;
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

                task.id            = task_ids[i];
                task.index         = i;
                task.tokens = std::move(tokenized_prompts[i]);

                slot_params ep;
                ep.embd_normalize = embd_normalize;
                task.params = ep;
                tasks.push_back(std::move(task));
            }

            for (int tid : task_ids) {
                ctx_server.queue_results.add_waiting_task_id(tid);
            }
            ctx_server.queue_tasks.post(std::move(tasks));
        }

        if (context->IsCancelled()) {
            for (int tid : task_ids) {
                ctx_server.queue_results.remove_waiting_task_id(tid);
            }
            return grpc::Status(grpc::StatusCode::CANCELLED, "Request cancelled by client");
        }

        // Collect responses (new ptr system)
        json responses = json::array();
        for (int tid : task_ids) {
            server_task_result_ptr res_ptr = nullptr;
            while (res_ptr == nullptr) {
                res_ptr = ctx_server.queue_results.recv_with_timeout(
                    std::unordered_set<int>{tid}, 1);
            }
            ctx_server.queue_results.remove_waiting_task_id(tid);
            if (res_ptr->is_error()) {
                json err_json = res_ptr->to_json();
                return grpc::Status(grpc::StatusCode::INTERNAL, err_json.value("message", "Error in receiving results"));
            }
            responses.push_back(res_ptr->to_json());
        }

        std::cout << "[DEBUG] Responses size: " << responses.size() << std::endl;

        // Process the responses and extract embeddings
        for (const auto & response_elem : responses) {
            // Check if the response has an "embedding" field
            if (response_elem.contains("embedding")) {
                json embedding_data = json_value(response_elem, "embedding", json::array());

                if (embedding_data.is_array() && !embedding_data.empty()) {
                    for (const auto & embedding_vector : embedding_data) {
                        if (embedding_vector.is_array()) {
                            for (const auto & embedding_value : embedding_vector) {
                                embeddingResult->add_embeddings(embedding_value.get<float>());
                            }
                        }
                    }
                }
            } else {
                // Check if the response itself contains the embedding data directly
                if (response_elem.is_array()) {
                    for (const auto & embedding_value : response_elem) {
                        embeddingResult->add_embeddings(embedding_value.get<float>());
                    }
                }
            }
        }

        return grpc::Status::OK;
    }

    grpc::Status Rerank(ServerContext* context, const backend::RerankRequest* request, backend::RerankResult* rerankResult) override {
        if (!params_base.embedding) {
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "This server does not support reranking. Start it with reranking options and without --embedding");
        }

        // Validate request
        if (request->query().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "\"query\" must be provided");
        }

        if (request->documents_size() == 0) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "\"documents\" must be a non-empty string array");
        }

        std::vector<std::string> documents;
        for (int i = 0; i < request->documents_size(); i++) {
            documents.push_back(request->documents(i));
        }

        // Tokenize query+document pairs as rerank tasks
        std::vector<int> task_ids;
        task_ids.reserve(documents.size());
        for (size_t i = 0; i < documents.size(); i++) {
            task_ids.push_back(ctx_server.queue_tasks.get_new_id());
        }

        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                // Tokenize as "query\ndocument" pair
                std::string combined = request->query() + "\n" + documents[i];
                auto tokens = tokenize_input_prompts(llama_model_get_vocab(ctx_server.model), ctx_server.mctx, combined, true, true);

                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id = task_ids[i];
                task.index = i;
                if (!tokens.empty()) {
                    task.tokens = std::move(tokens[0]);
                }
                tasks.push_back(std::move(task));
            }

            for (int tid : task_ids) {
                ctx_server.queue_results.add_waiting_task_id(tid);
            }
            ctx_server.queue_tasks.post(std::move(tasks));
        }

        if (context->IsCancelled()) {
            for (int tid : task_ids) {
                ctx_server.queue_results.remove_waiting_task_id(tid);
            }
            return grpc::Status(grpc::StatusCode::CANCELLED, "Request cancelled by client");
        }

        // Collect responses (new ptr system)
        json responses = json::array();
        for (int tid : task_ids) {
            server_task_result_ptr res_ptr = nullptr;
            while (res_ptr == nullptr) {
                res_ptr = ctx_server.queue_results.recv_with_timeout(
                    std::unordered_set<int>{tid}, 1);
            }
            ctx_server.queue_results.remove_waiting_task_id(tid);
            if (res_ptr->is_error()) {
                json err_json = res_ptr->to_json();
                return grpc::Status(grpc::StatusCode::INTERNAL, err_json.value("message", "Error in receiving results"));
            }
            responses.push_back(res_ptr->to_json());
        }

        // Sort responses by score in descending order
        std::sort(responses.begin(), responses.end(), [](const json& a, const json& b) {
            return a.value("score", 0.0f) > b.value("score", 0.0f);
        });

        // Crop results by request.top_n if specified
        int top_n = request->top_n();
        if (top_n > 0 && top_n < static_cast<int>(responses.size())) {
            responses = json(responses.begin(), responses.begin() + top_n);
        }
        // Set usage information
        backend::Usage* usage = rerankResult->mutable_usage();
        int total_tokens = 0;
        int prompt_tokens = 0;

        // Create document results
        for (const auto& response : responses) {
            backend::DocumentResult* doc_result = rerankResult->add_results();
            doc_result->set_index(response.value("index", 0));
            doc_result->set_text(request->documents(response.value("index", 0)));
            doc_result->set_relevance_score(response.value("score", 0.0f));

            int tokens_evaluated = response.value("tokens_evaluated", 0);
            total_tokens += tokens_evaluated;
            prompt_tokens += tokens_evaluated;
        }

        usage->set_total_tokens(total_tokens);
        usage->set_prompt_tokens(prompt_tokens);

        return grpc::Status::OK;
    }

    grpc::Status TokenizeString(ServerContext* /*context*/, const backend::PredictOptions* request, backend::TokenizationResponse* response) override {
        if (params_base.model.empty()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Model not loaded");
        }
        json body = parse_options(false, request, params_base, ctx_server.ctx);
        body["stream"] = false;

        if (body.count("prompt") != 0) {
            const bool add_special = json_value(body, "add_special", false);

            llama_tokens tokens = tokenize_mixed(llama_model_get_vocab(ctx_server.model), body.at("content"), add_special, true);

            for (const auto& token : tokens) {
                std::string piece = common_token_to_piece(ctx_server.ctx, token);
                response->add_tokens(token);
            }
        }

        return grpc::Status::OK;
    }

    grpc::Status GetMetrics(ServerContext* /*context*/, const backend::MetricsRequest* /*request*/, backend::MetricsResponse* response) override {

        // request metrics data using task queue
        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = task_id;
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        // get the result (new ptr system)
        server_task_result_ptr result_ptr = nullptr;
        while (result_ptr == nullptr) {
            result_ptr = ctx_server.queue_results.recv_with_timeout(
                std::unordered_set<int>{task_id}, 1);
        }
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result_ptr->is_error()) {
            response->set_slot_id(0);
            response->set_prompt_json_for_slot("");
            response->set_tokens_per_second(0);
            response->set_tokens_generated(0);
            response->set_prompt_tokens_processed(0);
            return grpc::Status(grpc::StatusCode::INTERNAL, "Error in receiving results");
        }

        // Extract metrics from result data
        json metrics_json = result_ptr->to_json();

        // Populate the response with metrics
        response->set_slot_id(0);
        response->set_prompt_json_for_slot("");

        double t_prompt_processing = metrics_json.value("t_prompt_processing", 0.0);
        uint64_t n_prompt_tokens_processed = metrics_json.value("n_prompt_tokens_processed", 0);
        response->set_tokens_per_second(t_prompt_processing > 0 ? 1.e3 / t_prompt_processing * n_prompt_tokens_processed : 0.);
        response->set_tokens_generated(metrics_json.value("n_tokens_predicted_total", 0));
        response->set_prompt_tokens_processed(metrics_json.value("n_prompt_tokens_processed_total", 0));

        return grpc::Status::OK;
    }

    grpc::Status ModelMetadata(ServerContext* /*context*/, const backend::ModelOptions* /*request*/, backend::ModelMetadataResponse* response) override {
        // Check if model is loaded
        if (params_base.model.empty()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Model not loaded");
        }

        // Check if chat templates are initialized
        if (ctx_server.chat_templates == nullptr) {
            response->set_supports_thinking(false);
            response->set_rendered_template("");
            return grpc::Status::OK;
        }

        // Detect thinking support using llama.cpp's function
        bool supports_thinking = common_chat_templates_support_enable_thinking(ctx_server.chat_templates.get());
        response->set_supports_thinking(supports_thinking);

        // Render the template with enable_thinking=true so Go code can detect thinking tokens
        std::string rendered_template = "";
        if (params_base.use_jinja) {
            common_chat_templates_inputs dummy_inputs;
            common_chat_msg msg;
            msg.role = "user";
            msg.content = "test";
            dummy_inputs.messages = {msg};
            dummy_inputs.enable_thinking = true;
            dummy_inputs.use_jinja = params_base.use_jinja;

            const auto rendered = common_chat_templates_apply(ctx_server.chat_templates.get(), dummy_inputs);
            rendered_template = rendered.prompt;
        }

        response->set_rendered_template(rendered_template);

        return grpc::Status::OK;
    }
};


int main(int argc, char** argv) {
  std::string server_address("localhost:50051");

  // Define long and short options
  struct option long_options[] = {
      {"addr", required_argument, nullptr, 'a'},
      {nullptr, 0, nullptr, 0}
  };

  // Parse command-line arguments
  int option;
  int option_index = 0;
  while ((option = getopt_long(argc, argv, "a:", long_options, &option_index)) != -1) {
    switch (option) {
      case 'a':
        server_address = optarg;
        break;
      default:
        std::cerr << "Usage: " << argv[0] << " [--addr=<address>] or [-a <address>]" << std::endl;
        return 1;
    }
  }

    server_context ctx_server;
    BackendServiceImpl service(ctx_server);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxMessageSize(50 * 1024 * 1024); // 50MB
    builder.SetMaxSendMessageSize(50 * 1024 * 1024); // 50MB
    builder.SetMaxReceiveMessageSize(50 * 1024 * 1024); // 50MB
    std::unique_ptr<Server> server(builder.BuildAndStart());
   // run the HTTP server in a thread - see comment below
    std::thread t([&]()
    {
        std::cout << "Server listening on " << server_address << std::endl;
        server->Wait();
        return 0;
    });

    // clean up function, to be called before exit
    auto clean_up = [&server, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        server->Shutdown();
        ctx_server.queue_tasks.terminate();
        llama_backend_free();
    };

    start_llama_server(ctx_server);
    std::cout << "stopping" << std::endl;

    clean_up();
    t.join();

    return 0;
}
