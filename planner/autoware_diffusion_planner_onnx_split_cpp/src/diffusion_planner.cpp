// Copyright 2025 TIER IV, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Diffusion Planner v3.0 - ONNX Split Architecture
// - context_encoder.onnx: Runs ONCE per planning cycle (v3: 6 encoder layers)
// - dit_core_dynamic.onnx: Runs N times in DPM-Solver loop (dynamic timestep)
// - Turn indicator: C++ linear layer 272 -> 5 classes (v3: added KEEP class)

#include "autoware/diffusion_planner_onnx_split/diffusion_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <filesystem>

// Simple JSON parsing for turn indicator weights
#include <sstream>

// === DEBUG LOGGING UTILITIES ===
namespace {
// Override at runtime with the DIFFUSION_PLANNER_DEBUG_DIR environment variable.
const std::string DEBUG_LOG_DIR = []() {
  const char * env = std::getenv("DIFFUSION_PLANNER_DEBUG_DIR");
  return std::string(env ? env : "/tmp/diffusion_planner_debug/");
}();
static int64_t g_frame_counter = 0;
static bool g_debug_enabled = true;

void ensure_debug_dir() {
  std::filesystem::create_directories(DEBUG_LOG_DIR);
}

void save_tensor_bin(const std::string& filename, const std::vector<float>& data) {
  std::ofstream f(DEBUG_LOG_DIR + filename, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

void save_tensor_bin(const std::string& filename, const std::vector<uint8_t>& data) {
  std::ofstream f(DEBUG_LOG_DIR + filename, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint8_t));
}

std::ofstream& get_summary_log() {
  static std::ofstream log;
  if (!log.is_open()) {
    ensure_debug_dir();
    log.open(DEBUG_LOG_DIR + "summary.log", std::ios::app);
  }
  return log;
}
}  // namespace
// === END DEBUG LOGGING UTILITIES ===

namespace autoware::diffusion_planner_onnx_split
{

// ============================================================================
// DiTInput Implementation
// ============================================================================

bool DiTInput::validate() const
{
  bool valid = true;

  // Expected sizes (matching context_encoder.onnx / dit_core_dynamic.onnx v3 requirements)
  auto check = [&](const char* name, size_t actual, size_t expected) {
    if (actual != expected) {
      std::cerr << "  " << name << ": " << actual << " (expected " << expected << ") MISMATCH!" << std::endl;
      valid = false;
    }
  };

  check("ego_agent_past", ego_agent_past.size(), 31 * 4);
  check("ego_current_state", ego_current_state.size(), 10);
  check("ego_shape", ego_shape.size(), 3);
  check("neighbor_agents_past", neighbor_agents_past.size(), 32 * 31 * 11);
  check("static_objects", static_objects.size(), 5 * 10);
  check("lanes", lanes.size(), 140 * 20 * 33);            // Model expects 140 segments
  check("lanes_speed_limit", lanes_speed_limit.size(), 140);  // Model expects 140
  check("lanes_has_speed_limit", lanes_has_speed_limit.size(), 140);  // Model expects 140
  check("route_lanes", route_lanes.size(), 25 * 20 * 33);
  check("route_lanes_speed_limit", route_lanes_speed_limit.size(), 25);
  check("route_lanes_has_speed_limit", route_lanes_has_speed_limit.size(), 25);
  check("polygons", polygons.size(), 10 * 40 * 2);
  check("line_strings", line_strings.size(), 10 * 20 * 2);
  check("goal_pose", goal_pose.size(), 4);
  check("turn_indicators", turn_indicators.size(), 31);
  check("neighbor_current_mask", neighbor_current_mask.size(), 32);

  return valid;
}

DiTInput DiTInput::create_zeros()
{
  DiTInput input;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 0.01f);

  auto fill_small_noise = [&](std::vector<float>& v, size_t size) {
    v.resize(size);
    for (size_t i = 0; i < size; ++i) {
      v[i] = dist(rng);
    }
  };

  fill_small_noise(input.ego_agent_past, 31 * 4);
  fill_small_noise(input.ego_current_state, 10);
  fill_small_noise(input.neighbor_agents_past, 32 * 31 * 11);
  fill_small_noise(input.static_objects, 5 * 10);
  fill_small_noise(input.lanes, 140 * 20 * 33);           // Model expects 140
  fill_small_noise(input.lanes_speed_limit, 140);         // Model expects 140
  fill_small_noise(input.route_lanes, 25 * 20 * 33);
  fill_small_noise(input.route_lanes_speed_limit, 25);
  fill_small_noise(input.polygons, 10 * 40 * 2);
  fill_small_noise(input.line_strings, 10 * 20 * 2);
  fill_small_noise(input.goal_pose, 4);
  fill_small_noise(input.turn_indicators, 31);

  input.ego_shape = {4.5f, 1.8f, 1.5f};
  input.lanes_has_speed_limit.resize(140, 0);
  input.route_lanes_has_speed_limit.resize(25, 0);
  input.neighbor_current_mask.resize(32, false);

  return input;
}

// ============================================================================
// VPNoiseSchedule Implementation
// ============================================================================

VPNoiseSchedule::VPNoiseSchedule(int num_steps, float beta_0, float beta_1)
: num_steps_(num_steps), beta_0_(beta_0), beta_1_(beta_1)
{
  const float t_start = 1.0f;
  const float t_end = 0.001f;

  float lambda_start = lambda(t_start);
  float lambda_end = lambda(t_end);

  timesteps_.resize(num_steps_ + 1);
  for (int i = 0; i <= num_steps_; ++i) {
    float lambda_i = lambda_start + (lambda_end - lambda_start) * i / num_steps_;
    timesteps_[i] = inverse_lambda(lambda_i);
  }

  std::cout << "[VPNoiseSchedule] Initialized with " << num_steps_ << " steps" << std::endl;
}

float VPNoiseSchedule::get_timestep(int step_index) const
{
  if (step_index < 0 || step_index > num_steps_) {
    throw std::out_of_range("Step index out of range");
  }
  return timesteps_[step_index];
}

float VPNoiseSchedule::log_alpha(float t) const
{
  return -0.25f * t * t * (beta_1_ - beta_0_) - 0.5f * t * beta_0_;
}

float VPNoiseSchedule::get_alpha(float t) const
{
  return std::exp(log_alpha(t));
}

float VPNoiseSchedule::get_sigma(float t) const
{
  float alpha = get_alpha(t);
  return std::sqrt(1.0f - alpha * alpha);
}

float VPNoiseSchedule::lambda(float t) const
{
  float log_a = log_alpha(t);
  float alpha_sq = std::exp(2.0f * log_a);
  float sigma_sq = 1.0f - alpha_sq;

  if (sigma_sq < 1e-10f) {
    return 10.0f;
  }

  return log_a - 0.5f * std::log(sigma_sq);
}

float VPNoiseSchedule::inverse_lambda(float lambda_val) const
{
  float t_low = 0.001f;
  float t_high = 1.0f;

  for (int iter = 0; iter < 50; ++iter) {
    float t_mid = (t_low + t_high) / 2.0f;
    float lambda_mid = lambda(t_mid);

    if (lambda_mid < lambda_val) {
      t_high = t_mid;
    } else {
      t_low = t_mid;
    }

    if (std::abs(t_high - t_low) < 1e-6f) {
      break;
    }
  }

  return (t_low + t_high) / 2.0f;
}

// ============================================================================
// DiffusionPlanner Implementation
// ============================================================================

DiffusionPlanner::DiffusionPlanner(const DiffusionPlannerConfig& config)
: config_(config)
, rng_(std::random_device{}())
, normal_dist_(0.0f, 1.0f)
{
  std::cout << "========================================" << std::endl;
  std::cout << "  DiffusionPlanner (GraphSurgeon Split)" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Encoder: " << config_.encoder_model_path << std::endl;
  std::cout << "DiT Core: " << config_.dit_model_path << std::endl;
  std::cout << "Turn Indicator Weights: " << config_.turn_indicator_weights_path << std::endl;
  std::cout << "Inference steps: " << config_.num_inference_steps << std::endl;
  std::cout << "Use GPU: " << (config_.use_gpu ? "Yes" : "No") << std::endl;
  std::cout << "========================================" << std::endl;

  // Apply TurnIndicatorManager config
  turn_indicator_keep_offset_ = config_.turn_indicator_keep_offset;
  turn_indicator_hold_duration_ns_ =
    static_cast<int64_t>(config_.turn_indicator_hold_duration * 1e9);

  init_sessions();
  load_turn_indicator_weights();

  noise_schedule_ = std::make_unique<VPNoiseSchedule>(
    config_.num_inference_steps,
    config_.beta_0,
    config_.beta_1);

  initialized_ = true;
  std::cout << "DiffusionPlanner initialized successfully!" << std::endl;
}

DiffusionPlanner::~DiffusionPlanner() = default;

void DiffusionPlanner::init_sessions()
{
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DiffusionPlannerSplit");

  session_options_ = std::make_unique<Ort::SessionOptions>();
  session_options_->SetIntraOpNumThreads(4);
  // Use ORT_ENABLE_BASIC to avoid FusedMatMul optimization crash with all-zero tensors
  session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

  if (config_.use_gpu) {
    try {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = config_.gpu_device_id;
      session_options_->AppendExecutionProvider_CUDA(cuda_options);
      std::cout << "CUDA execution provider enabled" << std::endl;
    } catch (const Ort::Exception& e) {
      std::cerr << "Warning: Failed to enable CUDA: " << e.what() << std::endl;
    }
  }

  memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::AllocatorWithDefaultOptions allocator;

  // Load Encoder session
  std::cout << "Loading context_encoder.onnx (v3)..." << std::endl;
  encoder_session_ = std::make_unique<Ort::Session>(
    *env_, config_.encoder_model_path.c_str(), *session_options_);

  size_t num_enc_inputs = encoder_session_->GetInputCount();
  std::cout << "Encoder inputs (" << num_enc_inputs << "):" << std::endl;
  for (size_t i = 0; i < num_enc_inputs; ++i) {
    auto name = encoder_session_->GetInputNameAllocated(i, allocator);
    encoder_input_names_.push_back(strdup(name.get()));
    std::cout << "  [" << i << "] " << encoder_input_names_.back() << std::endl;
  }

  size_t num_enc_outputs = encoder_session_->GetOutputCount();
  for (size_t i = 0; i < num_enc_outputs; ++i) {
    auto name = encoder_session_->GetOutputNameAllocated(i, allocator);
    encoder_output_names_.push_back(strdup(name.get()));
  }
  std::cout << "Encoder output: " << encoder_output_names_[0] << std::endl;

  // Load DiT Core session
  std::cout << "Loading dit_core_dynamic.onnx (v3)..." << std::endl;
  dit_session_ = std::make_unique<Ort::Session>(
    *env_, config_.dit_model_path.c_str(), *session_options_);

  size_t num_dit_inputs = dit_session_->GetInputCount();
  std::cout << "DiT inputs (" << num_dit_inputs << "):" << std::endl;
  for (size_t i = 0; i < num_dit_inputs; ++i) {
    auto name = dit_session_->GetInputNameAllocated(i, allocator);
    dit_input_names_.push_back(strdup(name.get()));
    std::cout << "  [" << i << "] " << dit_input_names_.back() << std::endl;
  }

  // Verify dit_core_dynamic.onnx input order (v3)
  // Expected order: x, timestep, context_embedding, neighbor_current_mask
  const std::vector<std::string> expected_dit_inputs = {
    "x", "timestep", "context_embedding", "neighbor_current_mask"
  };

  if (num_dit_inputs != expected_dit_inputs.size()) {
    std::cerr << "WARNING: dit_core.onnx has " << num_dit_inputs
              << " inputs, expected " << expected_dit_inputs.size() << std::endl;
  } else {
    bool order_ok = true;
    for (size_t i = 0; i < num_dit_inputs; ++i) {
      if (std::string(dit_input_names_[i]) != expected_dit_inputs[i]) {
        std::cerr << "WARNING: dit_core input[" << i << "] is '" << dit_input_names_[i]
                  << "', expected '" << expected_dit_inputs[i] << "'" << std::endl;
        order_ok = false;
      }
    }
    if (order_ok) {
      std::cout << "DiT input order verified OK" << std::endl;
    } else {
      std::cerr << "WARNING: DiT input order mismatch - may cause incorrect results!" << std::endl;
    }
  }

  size_t num_dit_outputs = dit_session_->GetOutputCount();
  for (size_t i = 0; i < num_dit_outputs; ++i) {
    auto name = dit_session_->GetOutputNameAllocated(i, allocator);
    dit_output_names_.push_back(strdup(name.get()));
  }
  std::cout << "DiT output: " << dit_output_names_[0] << std::endl;
}

void DiffusionPlanner::load_turn_indicator_weights()
{
  std::cout << "Loading turn indicator weights..." << std::endl;

  std::ifstream file(config_.turn_indicator_weights_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open turn indicator weights file");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();

  // Simple JSON parsing for our known format
  // Format: {"weight": [[...], [...], [...], [...]], "bias": [...]}
  auto parse_array = [](const std::string& json, const std::string& key) -> std::vector<float> {
    std::vector<float> result;
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;

    size_t start = json.find('[', key_pos);
    if (start == std::string::npos) return result;

    // Check if this is a nested array (weight) or flat array (bias)
    bool is_nested = (start + 1 < json.size() && json[start + 1] == '[');

    // Start bracket count at 1 because we're entering the outer bracket
    int bracket_count = 1;
    std::string num_str;

    for (size_t i = start + 1; i < json.size(); ++i) {
      char c = json[i];
      if (c == '[') {
        bracket_count++;
      } else if (c == ']') {
        if (!num_str.empty()) {
          result.push_back(std::stof(num_str));
          num_str.clear();
        }
        bracket_count--;
        if (bracket_count == 0) break;  // Exit when we close the outer bracket
      } else if (c == ',' || c == ' ' || c == '\n' || c == '\t') {
        if (!num_str.empty()) {
          result.push_back(std::stof(num_str));
          num_str.clear();
        }
      } else if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+') {
        num_str += c;
      }
    }

    return result;
  };

  turn_indicator_weight_ = parse_array(json, "weight");
  turn_indicator_bias_ = parse_array(json, "bias");

  std::cout << "  Weight size: " << turn_indicator_weight_.size()
            << " (expected: " << 5 * 272 << ")" << std::endl;
  std::cout << "  Bias size: " << turn_indicator_bias_.size()
            << " (expected: 5)" << std::endl;

  if (turn_indicator_weight_.size() != 5 * 272 || turn_indicator_bias_.size() != 5) {
    throw std::runtime_error("Invalid turn indicator weights dimensions (expected 5x272 for v3)");
  }
}

std::vector<float> DiffusionPlanner::sample_initial_noise()
{
  const size_t size = NUM_SAMPLES * SEQUENCE_LENGTH * STATE_DIM;
  // Sample from standard normal distribution N(0, 1) as required by
  // diffusion models. The DPM-Solver++ denoising loop starts from pure
  // noise and iteratively refines it into a clean trajectory prediction.
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> noise(size);
  for (size_t i = 0; i < size; ++i) {
    noise[i] = dist(rng_);
  }
  return noise;
}

std::vector<Ort::Value> DiffusionPlanner::create_encoder_inputs(const DiTInput& input)
{
  std::vector<Ort::Value> tensors;

  auto create_tensor = [this](const std::vector<float>& data, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(data.data()), data.size(),
      shape.data(), shape.size());
  };

  // Encoder expects 16 inputs (same as original model minus timestep which doesn't exist in encoder)
  // The encoder will ignore sampled_trajectories since it doesn't need it
  // Order based on model inspection

  // Static buffer for sampled_trajectories (encoder might need shape info)
  static std::vector<float> dummy_sampled(BATCH_SIZE * NUM_SAMPLES * SEQUENCE_LENGTH * STATE_DIM, 0.0f);

  // Boolean buffers
  static std::vector<bool> lanes_has_speed_limit_bool(140);
  static std::vector<bool> route_lanes_has_speed_limit_bool(25);
  for (size_t i = 0; i < 140 && i < input.lanes_has_speed_limit.size(); ++i) {
    lanes_has_speed_limit_bool[i] = (input.lanes_has_speed_limit[i] != 0);
  }
  for (size_t i = 0; i < 25 && i < input.route_lanes_has_speed_limit.size(); ++i) {
    route_lanes_has_speed_limit_bool[i] = (input.route_lanes_has_speed_limit[i] != 0);
  }

  // Create tensors in order expected by encoder
  // Based on the model's input names
  tensors.push_back(create_tensor(dummy_sampled, {BATCH_SIZE, NUM_SAMPLES, SEQUENCE_LENGTH, STATE_DIM}));
  tensors.push_back(create_tensor(input.ego_agent_past, {BATCH_SIZE, 31, 4}));
  tensors.push_back(create_tensor(input.ego_current_state, {BATCH_SIZE, 10}));
  tensors.push_back(create_tensor(input.neighbor_agents_past, {BATCH_SIZE, 32, 31, 11}));
  tensors.push_back(create_tensor(input.static_objects, {BATCH_SIZE, 5, 10}));
  tensors.push_back(create_tensor(input.lanes, {BATCH_SIZE, 140, 20, 33}));
  tensors.push_back(create_tensor(input.lanes_speed_limit, {BATCH_SIZE, 140, 1}));

  // Bool tensor for lanes_has_speed_limit
  {
    std::vector<int64_t> shape = {BATCH_SIZE, 140, 1};
    static bool lanes_bool_arr[140];
    for (int i = 0; i < 140; ++i) lanes_bool_arr[i] = lanes_has_speed_limit_bool[i];
    tensors.push_back(Ort::Value::CreateTensor<bool>(
      memory_info_, lanes_bool_arr, 140, shape.data(), shape.size()));
  }

  tensors.push_back(create_tensor(input.route_lanes, {BATCH_SIZE, 25, 20, 33}));
  tensors.push_back(create_tensor(input.route_lanes_speed_limit, {BATCH_SIZE, 25, 1}));

  // Bool tensor for route_lanes_has_speed_limit
  {
    std::vector<int64_t> shape = {BATCH_SIZE, 25, 1};
    static bool route_bool_arr[25];
    for (int i = 0; i < 25; ++i) route_bool_arr[i] = route_lanes_has_speed_limit_bool[i];
    tensors.push_back(Ort::Value::CreateTensor<bool>(
      memory_info_, route_bool_arr, 25, shape.data(), shape.size()));
  }

  tensors.push_back(create_tensor(input.polygons, {BATCH_SIZE, 10, 40, 2}));
  tensors.push_back(create_tensor(input.line_strings, {BATCH_SIZE, 10, 20, 2}));
  tensors.push_back(create_tensor(input.goal_pose, {BATCH_SIZE, 4}));
  tensors.push_back(create_tensor(input.ego_shape, {BATCH_SIZE, 3}));
  tensors.push_back(create_tensor(input.turn_indicators, {BATCH_SIZE, 31}));

  return tensors;
}

void DiffusionPlanner::run_encoder(const DiTInput& input)
{
  // With ORT_ENABLE_BASIC optimization level, all-zero neighbor data is handled correctly
  // (the FusedMatMul optimization that caused crashes is only in ORT_ENABLE_ALL)
  auto input_tensors = create_encoder_inputs(input);

  auto output_tensors = encoder_session_->Run(
    Ort::RunOptions{nullptr},
    encoder_input_names_.data(), input_tensors.data(), input_tensors.size(),
    encoder_output_names_.data(), encoder_output_names_.size());

  // Extract context_embedding: expected [batch, 226, 256]
  auto* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  size_t output_size = 1;
  for (auto dim : output_shape) {
    output_size *= dim;
  }

  // Cache context_embedding: v3 shape is [1, 226, 256] (batch-first)
  cached_context_embedding_.assign(output_data, output_data + output_size);

  // Cache neighbor_current_mask for DiT loop
  cached_neighbor_current_mask_ = input.neighbor_current_mask;

  // Compute mean pooling for turn indicator: mean over seq dim -> [256]
  // Shape: [1, 226, 256] — iterate over 226 tokens
  cached_encoding_pooled_.resize(EMBED_DIM, 0.0f);
  std::fill(cached_encoding_pooled_.begin(), cached_encoding_pooled_.end(), 0.0f);
  for (int s = 0; s < CONTEXT_SEQ_LEN; ++s) {
    for (int d = 0; d < EMBED_DIM; ++d) {
      cached_encoding_pooled_[d] += cached_context_embedding_[s * EMBED_DIM + d];
    }
  }
  for (int d = 0; d < EMBED_DIM; ++d) {
    cached_encoding_pooled_[d] /= static_cast<float>(CONTEXT_SEQ_LEN);
  }

  std::cout << "[Encoder] Context embedding cached: " << cached_context_embedding_.size()
            << " elements [1,226,256], pooled: " << cached_encoding_pooled_.size() << std::endl;
}

std::vector<Ort::Value> DiffusionPlanner::create_dit_inputs(
  const std::vector<float>& x_flat,
  float timestep)
{
  // dit_core_dynamic.onnx inputs (v3):
  //   x                    [1, 33, 324]  (flattened: 81*4=324)
  //   timestep             [1]
  //   context_embedding    [1, 226, 256] (batch-first in v3)
  //   neighbor_current_mask [1, 32]      (bool)

  std::vector<Ort::Value> tensors;

  auto create_float_tensor = [this](const std::vector<float>& data, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(data.data()), data.size(),
      shape.data(), shape.size());
  };

  // x: [1, 33, 324]
  tensors.push_back(create_float_tensor(x_flat, {BATCH_SIZE, NUM_SAMPLES, DIT_X_DIM}));

  // timestep: [1]
  static std::vector<float> timestep_data(1);
  timestep_data[0] = timestep;
  tensors.push_back(create_float_tensor(timestep_data, {1}));

  // context_embedding: [1, 226, 256] — cached as batch-first in v3
  tensors.push_back(create_float_tensor(
    cached_context_embedding_, {BATCH_SIZE, CONTEXT_SEQ_LEN, EMBED_DIM}));

  // neighbor_current_mask: [1, 32] bool
  // Must convert std::vector<bool> to bool array (ORT requires contiguous bool*)
  static std::vector<bool> mask_buf(NUM_NEIGHBORS);
  mask_buf = cached_neighbor_current_mask_;
  static bool mask_arr[32];
  for (int i = 0; i < NUM_NEIGHBORS; ++i) mask_arr[i] = mask_buf[i];
  const std::vector<int64_t> mask_shape = {BATCH_SIZE, NUM_NEIGHBORS};
  tensors.push_back(Ort::Value::CreateTensor<bool>(
    memory_info_, mask_arr, NUM_NEIGHBORS, mask_shape.data(), mask_shape.size()));

  return tensors;
}

std::vector<float> DiffusionPlanner::run_dit_step(
  const std::vector<float>& noisy_trajectory,
  float timestep)
{
  auto input_tensors = create_dit_inputs(noisy_trajectory, timestep);

  auto output_tensors = dit_session_->Run(
    Ort::RunOptions{nullptr},
    dit_input_names_.data(), input_tensors.data(), input_tensors.size(),
    dit_output_names_.data(), dit_output_names_.size());

  // Output: dit_prediction [1, 33, 324] where 324 = 81*4 (same as v2)
  auto* output_data = output_tensors[0].GetTensorMutableData<float>();
  const size_t output_size = NUM_SAMPLES * DIT_X_DIM;  // 33 * 324

  // Return as flat [33, 81, 4] for DPM-Solver compatibility
  std::vector<float> prediction(NUM_SAMPLES * SEQUENCE_LENGTH * STATE_DIM);
  for (int agent = 0; agent < NUM_SAMPLES; ++agent) {
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
      for (int d = 0; d < STATE_DIM; ++d) {
        prediction[(agent * SEQUENCE_LENGTH + t) * STATE_DIM + d] =
          output_data[agent * DIT_X_DIM + t * STATE_DIM + d];
      }
    }
  }

  return prediction;
}

std::vector<float> DiffusionPlanner::compute_turn_indicator(const std::vector<float>& x0)
{
  // Turn indicator (v3): Linear 272 -> 5 classes
  // Input: [ego_traj_xy(16), encoding_pooled(256)] -> [272]
  // Ego trajectory: x0[:, 0, 1::10, :2] -> timesteps 1,11,21,31,41,51,61,71

  std::vector<float> ego_traj(16);
  for (int i = 0; i < 8; ++i) {
    const int t_idx = 1 + i * 10;
    const size_t src = (0 * SEQUENCE_LENGTH + t_idx) * STATE_DIM;
    ego_traj[i * 2 + 0] = x0[src + 0];
    ego_traj[i * 2 + 1] = x0[src + 1];
  }

  std::vector<float> feat(272);
  std::copy(ego_traj.begin(), ego_traj.end(), feat.begin());
  std::copy(cached_encoding_pooled_.begin(), cached_encoding_pooled_.end(), feat.begin() + 16);

  // W: [5, 272], b: [5]
  std::vector<float> logit(TURN_INDICATOR_CLASSES);
  for (int i = 0; i < TURN_INDICATOR_CLASSES; ++i) {
    float sum = turn_indicator_bias_[i];
    for (int j = 0; j < 272; ++j) {
      sum += feat[j] * turn_indicator_weight_[i * 272 + j];
    }
    logit[i] = sum;
  }
  return logit;
}

uint8_t DiffusionPlanner::evaluate_turn_indicator(
  const std::vector<float>& logit,
  int64_t stamp_ns,
  uint8_t prev_report)
{
  // TurnIndicatorManager logic (mirrors Tier IV v3 C++ implementation):
  // 1. If within hold_duration of last non-KEEP command, hold that command
  // 2. Add keep_offset to KEEP logit, then softmax + argmax
  // 3. If KEEP selected, return prev_report; else return predicted class

  // Hold duration check
  if (last_non_keep_stamp_ns_ > 0) {
    const int64_t expiry = last_non_keep_stamp_ns_ + turn_indicator_hold_duration_ns_;
    if (stamp_ns <= expiry) {
      return static_cast<uint8_t>(last_non_keep_command_);
    }
  }

  // Apply keep_offset, then softmax
  std::vector<float> adj = logit;
  adj[TURN_INDICATOR_KEEP] += turn_indicator_keep_offset_;

  const float max_logit = *std::max_element(adj.begin(), adj.end());
  std::vector<float> prob(TURN_INDICATOR_CLASSES);
  float sum = 1e-4f;
  for (int i = 0; i < TURN_INDICATOR_CLASSES; ++i) {
    prob[i] = std::exp(adj[i] - max_logit);
    sum += prob[i];
  }
  for (auto& p : prob) p /= sum;

  const int max_idx = std::distance(prob.begin(), std::max_element(prob.begin(), prob.end()));
  const bool keep_selected = (max_idx == TURN_INDICATOR_KEEP);
  const uint8_t cmd = keep_selected ? prev_report : static_cast<uint8_t>(max_idx);

  if (!keep_selected) {
    last_non_keep_command_ = cmd;
    last_non_keep_stamp_ns_ = stamp_ns;
  }
  return cmd;
}

void DiffusionPlanner::apply_initial_state_constraint(
  std::vector<float>& x_t,
  const std::vector<float>& current_states)
{
  // Apply constraint: set t=0 timestep to current_states for each agent
  // x_t shape: [NUM_SAMPLES, SEQUENCE_LENGTH, STATE_DIM] flattened
  // current_states shape: [NUM_SAMPLES, STATE_DIM] flattened

  for (size_t agent = 0; agent < NUM_SAMPLES; ++agent) {
    size_t x_t_base = agent * SEQUENCE_LENGTH * STATE_DIM;  // First timestep (t=0)
    size_t cs_base = agent * STATE_DIM;

    for (size_t d = 0; d < STATE_DIM; ++d) {
      x_t[x_t_base + d] = current_states[cs_base + d];
    }
  }
}

void DiffusionPlanner::dpm_solver_first_order_update(
  std::vector<float>& x_t,
  const std::vector<float>& model_output,
  float t_current,
  float t_next)
{
  float sigma_s = noise_schedule_->get_sigma(t_current);
  float alpha_t = noise_schedule_->get_alpha(t_next);
  float sigma_t = noise_schedule_->get_sigma(t_next);

  float lambda_s = noise_schedule_->lambda(t_current);
  float lambda_t = noise_schedule_->lambda(t_next);
  float h = lambda_t - lambda_s;

  float coef_x = sigma_t / sigma_s;
  float coef_model = alpha_t * (1.0f - std::exp(-h));

  for (size_t i = 0; i < x_t.size() && i < model_output.size(); ++i) {
    x_t[i] = coef_x * x_t[i] + coef_model * model_output[i];
  }
}

void DiffusionPlanner::dpm_solver_second_order_update(
  std::vector<float>& x_t,
  const std::vector<float>& model_output_0,
  const std::vector<float>& model_output_1,
  float t_prev,
  float t_current,
  float t_next)
{
  float lambda_prev = noise_schedule_->lambda(t_prev);
  float lambda_s = noise_schedule_->lambda(t_current);
  float lambda_t = noise_schedule_->lambda(t_next);

  float alpha_t = noise_schedule_->get_alpha(t_next);
  float sigma_s = noise_schedule_->get_sigma(t_current);
  float sigma_t = noise_schedule_->get_sigma(t_next);

  float h = lambda_t - lambda_s;
  float h_prev = lambda_s - lambda_prev;
  float r = h_prev / h;

  float coef_x = sigma_t / sigma_s;
  float coef_model = alpha_t * (1.0f - std::exp(-h));
  float coef_correction = 0.5f * coef_model;

  for (size_t i = 0; i < x_t.size(); ++i) {
    float D1 = (model_output_0[i] - model_output_1[i]) / r;
    x_t[i] = coef_x * x_t[i] + coef_model * model_output_0[i] + coef_correction * D1;
  }
}

void DiffusionPlanner::denormalize_trajectory(std::vector<float>& trajectory)
{
  for (size_t sample = 0; sample < NUM_SAMPLES; ++sample) {
    for (size_t t = 0; t < SEQUENCE_LENGTH; ++t) {
      size_t base = (sample * SEQUENCE_LENGTH + t) * STATE_DIM;

      // Apply denormalization: x = x_normalized * std + mean
      trajectory[base + 0] = trajectory[base + 0] * NORM_STD_X + NORM_MEAN_X;  // x: mean=10.0
      trajectory[base + 1] = trajectory[base + 1] * NORM_STD_Y + NORM_MEAN_Y;  // y: mean=0.0
      trajectory[base + 2] = std::clamp(trajectory[base + 2], -1.0f, 1.0f);    // cos(yaw)
      trajectory[base + 3] = std::clamp(trajectory[base + 3], -1.0f, 1.0f);    // sin(yaw)
    }
  }
}

PlannerOutput DiffusionPlanner::plan(const DiTInput& input)
{
  PlannerOutput output;
  output.success = false;

  if (!initialized_) {
    std::cerr << "[DiffusionPlanner] Not initialized!" << std::endl;
    return output;
  }

  if (!input.validate()) {
    std::cerr << "[DiffusionPlanner] Invalid input dimensions!" << std::endl;
    return output;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    // Step 1: Run encoder ONCE
    std::cout << "[DiffusionPlanner] Running encoder (once)..." << std::endl;
    run_encoder(input);

    // === DEBUG: Save input data ===
    if (g_debug_enabled) {
      ensure_debug_dir();
      std::string frame_id = "frame_" + std::to_string(g_frame_counter);

      // Save input tensors
      save_tensor_bin(frame_id + "_ego_current_state.bin", input.ego_current_state);
      save_tensor_bin(frame_id + "_ego_agent_past.bin", input.ego_agent_past);
      save_tensor_bin(frame_id + "_neighbor_agents_past.bin", input.neighbor_agents_past);
      save_tensor_bin(frame_id + "_goal_pose.bin", input.goal_pose);
      save_tensor_bin(frame_id + "_lanes.bin", input.lanes);
      save_tensor_bin(frame_id + "_route_lanes.bin", input.route_lanes);
      save_tensor_bin(frame_id + "_context_embedding.bin", cached_context_embedding_);

      auto& log = get_summary_log();
      log << "=== Frame " << g_frame_counter << " ===" << std::endl;
      log << "ego_current_state[0:3]: " << input.ego_current_state[0] << ", "
          << input.ego_current_state[1] << ", " << input.ego_current_state[2] << std::endl;
      log << "goal_pose: " << input.goal_pose[0] << ", " << input.goal_pose[1] << ", "
          << input.goal_pose[2] << ", " << input.goal_pose[3] << std::endl;
      log << "context_embedding[0:3]: " << cached_context_embedding_[0] << ", "
          << cached_context_embedding_[1] << ", " << cached_context_embedding_[2] << std::endl;
    }

    // Step 2: Build current_states for initial_state_constraint
    // current_states = [ego_current (4), neighbor_0_current (4), ..., neighbor_31_current (4)]
    // Shape: [NUM_SAMPLES=33, STATE_DIM=4]
    std::vector<float> current_states(NUM_SAMPLES * STATE_DIM, 0.0f);

    // Ego current state (first 4 elements of ego_current_state, already normalized)
    for (size_t d = 0; d < STATE_DIM; ++d) {
      current_states[d] = input.ego_current_state[d];
    }

    // Neighbor current states (last timestep of each neighbor, first 4 elements)
    // neighbor_agents_past shape: [32, 31, 11]
    constexpr size_t NEIGHBOR_TIME_LEN = 31;
    constexpr size_t NEIGHBOR_STATE_DIM = 11;
    constexpr size_t NEIGHBOR_NUM = 32;

    for (size_t n = 0; n < NEIGHBOR_NUM; ++n) {
      // Last timestep (index 30) of neighbor n, first 4 elements
      size_t neighbor_base = n * NEIGHBOR_TIME_LEN * NEIGHBOR_STATE_DIM + 30 * NEIGHBOR_STATE_DIM;
      size_t cs_base = (n + 1) * STATE_DIM;  // +1 because ego is at index 0

      for (size_t d = 0; d < STATE_DIM; ++d) {
        current_states[cs_base + d] = input.neighbor_agents_past[neighbor_base + d];
      }
    }

    // Step 3: Sample initial noise
    std::cout << "[DiffusionPlanner] Sampling initial noise..." << std::endl;
    std::vector<float> x_t = sample_initial_noise();

    // Apply initial state constraint to noise
    apply_initial_state_constraint(x_t, current_states);

    // Step 4: DPM-Solver++ denoising loop with anytime support
    auto cycle_start = std::chrono::high_resolution_clock::now();

    // Simulate upstream compute jitter (controlled by anytime_log_csv as jitter switch)
    float jitter_ms = 0.0f;
    if (config_.anytime_log_csv) {
      // Random jitter between margin_ms and budget_ms to simulate perception pipeline variance
      std::uniform_real_distribution<float> jitter_dist(
        config_.anytime_margin_ms, config_.anytime_budget_ms);  // reuse as jitter_min/max
      jitter_ms = jitter_dist(rng_);
      // Busy-wait to simulate upstream delay
      auto jitter_end = cycle_start + std::chrono::microseconds(static_cast<int>(jitter_ms * 1000));
      while (std::chrono::high_resolution_clock::now() < jitter_end) {}
    }

    auto loop_start = std::chrono::high_resolution_clock::now();

    // Budget-aware step selection
    int effective_steps = config_.num_inference_steps;
    const float cycle_budget_ms = 100.0f;  // 10Hz hard deadline
    const float dit_estimate_ms = 6.0f;    // estimated per-DiT-step cost

    if (config_.anytime_enabled) {
      float elapsed = std::chrono::duration<float, std::milli>(loop_start - cycle_start).count();
      float remaining = cycle_budget_ms - elapsed - dit_estimate_ms;  // reserve for denoise-to-zero
      int max_n = std::max(2, static_cast<int>(remaining / dit_estimate_ms));
      effective_steps = std::min(config_.num_inference_steps, max_n);
      // Rebuild noise schedule for the selected N
      noise_schedule_ = std::make_unique<VPNoiseSchedule>(effective_steps, config_.beta_0, config_.beta_1);
    }

    std::vector<float> model_prev;
    std::vector<float> last_x0_prediction;
    std::vector<float> step_latencies_ms;
    std::vector<float> step_deltas;
    int actual_steps = effective_steps;
    std::string stop_reason = "completed";

    for (int step = 0; step < effective_steps; ++step) {
      float t_current = noise_schedule_->get_timestep(step);
      float t_next = noise_schedule_->get_timestep(step + 1);

      auto step_start = std::chrono::high_resolution_clock::now();
      // v3 DiT expects x flattened as [1, 33, 324]; x_t is stored as [33, 81, 4]
      // Memory layout is identical (both row-major), so direct use is valid.
      auto x0_prediction = run_dit_step(x_t, t_current);
      auto step_end = std::chrono::high_resolution_clock::now();
      float step_ms = std::chrono::duration<float, std::milli>(step_end - step_start).count();
      step_latencies_ms.push_back(step_ms);

      // Compute convergence delta
      float delta = 0.0f;
      if (!model_prev.empty()) {
        delta = compute_convergence_delta(x0_prediction, model_prev);
      }
      step_deltas.push_back(delta);

      // Anytime: convergence early stop (after min 2 steps)
      if (config_.anytime_enabled && step >= 2 && !model_prev.empty()) {
        if (delta < config_.anytime_convergence_threshold) {
          actual_steps = step + 1;
          last_x0_prediction = x0_prediction;
          stop_reason = "converged";
          // Still do the solver update for this step
          if (step == 0 || model_prev.empty()) {
            dpm_solver_first_order_update(x_t, x0_prediction, t_current, t_next);
          } else {
            float t_prev = noise_schedule_->get_timestep(step - 1);
            dpm_solver_second_order_update(x_t, x0_prediction, model_prev, t_prev, t_current, t_next);
          }
          apply_initial_state_constraint(x_t, current_states);
          break;
        }
      }

      // Anytime: deadline guard
      if (config_.anytime_enabled && step < effective_steps - 1) {
        float total_elapsed = std::chrono::duration<float, std::milli>(
          std::chrono::high_resolution_clock::now() - cycle_start).count();
        float needed = dit_estimate_ms + 3.0f;  // next step + margin
        if (total_elapsed + needed > cycle_budget_ms) {
          actual_steps = step + 1;
          last_x0_prediction = x0_prediction;
          stop_reason = "deadline";
          if (step == 0 || model_prev.empty()) {
            dpm_solver_first_order_update(x_t, x0_prediction, t_current, t_next);
          } else {
            float t_prev = noise_schedule_->get_timestep(step - 1);
            dpm_solver_second_order_update(x_t, x0_prediction, model_prev, t_prev, t_current, t_next);
          }
          apply_initial_state_constraint(x_t, current_states);
          break;
        }
      }

      if (step == 0 || model_prev.empty()) {
        dpm_solver_first_order_update(x_t, x0_prediction, t_current, t_next);
      } else {
        float t_prev = noise_schedule_->get_timestep(step - 1);
        dpm_solver_second_order_update(x_t, x0_prediction, model_prev, t_prev, t_current, t_next);
      }

      apply_initial_state_constraint(x_t, current_states);
      model_prev = x0_prediction;
      last_x0_prediction = x0_prediction;
    }

    // denoise_to_zero or use last prediction
    float dtz_ms = 0.0f;
    if (stop_reason == "completed") {
      float t_0 = noise_schedule_->get_timestep(effective_steps);
      auto dtz_start = std::chrono::high_resolution_clock::now();
      x_t = run_dit_step(x_t, t_0);
      dtz_ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - dtz_start).count();
    } else if (last_x0_prediction.size() > 0) {
      x_t = last_x0_prediction;
    }

    auto loop_end = std::chrono::high_resolution_clock::now();
    float loop_ms = std::chrono::duration<float, std::milli>(loop_end - loop_start).count();
    float total_cycle_ms = std::chrono::duration<float, std::milli>(loop_end - cycle_start).count();

    // Populate anytime logging fields in output (for node-level CSV with ego pose)
    output.anytime_jitter_ms = jitter_ms;
    output.anytime_planned_steps = effective_steps;
    output.anytime_actual_steps = actual_steps;
    output.anytime_denoise_ms = loop_ms;
    output.anytime_total_cycle_ms = total_cycle_ms;
    output.anytime_stop_reason = stop_reason;
    output.anytime_per_step_delta = step_deltas;

    std::cout << "[DiffusionPlanner] " << actual_steps << " steps, "
              << loop_ms << "ms denoise, " << total_cycle_ms << "ms total"
              << (jitter_ms > 0 ? " (jitter=" + std::to_string((int)jitter_ms) + "ms)" : "")
              << " [" << stop_reason << "]" << std::endl;

    // Apply constraint one final time
    apply_initial_state_constraint(x_t, current_states);

    // Compute turn indicator BEFORE denormalization (weights trained on normalized coords)
    auto turn_logit = compute_turn_indicator(x_t);

    // Step 5: Denormalize output
    denormalize_trajectory(x_t);

    // Step 6: Extract trajectories
    output.raw_prediction.resize(NUM_SAMPLES * FUTURE_LENGTH * STATE_DIM);
    for (int agent = 0; agent < NUM_SAMPLES; ++agent) {
      for (int t = 0; t < FUTURE_LENGTH; ++t) {
        size_t src_idx = (agent * SEQUENCE_LENGTH + t + 1) * STATE_DIM;
        size_t dst_idx = (agent * FUTURE_LENGTH + t) * STATE_DIM;
        for (int d = 0; d < STATE_DIM; ++d) {
          output.raw_prediction[dst_idx + d] = x_t[src_idx + d];
        }
      }
    }

    // v3: 5-class turn indicator logits
    output.turn_indicator_logit = turn_logit;  // [5]: NONE/DISABLE/LEFT/RIGHT/KEEP

    // === DEBUG: Save output predictions and turn indicator ===
    if (g_debug_enabled) {
      std::string frame_id = "frame_" + std::to_string(g_frame_counter);

      // Save raw_prediction [33, 80, 4]
      save_tensor_bin(frame_id + "_output.bin", output.raw_prediction);

      // Save turn indicator [4]
      save_tensor_bin(frame_id + "_turn_indicator.bin", turn_logit);

      auto& log = get_summary_log();
      log << "output size: " << output.raw_prediction.size() << std::endl;

      // ego t=0 (first 4 values)
      log << "ego traj t=0: " << output.raw_prediction[0] << ", "
          << output.raw_prediction[1] << ", " << output.raw_prediction[2] << ", "
          << output.raw_prediction[3] << std::endl;

      // ego t=79 (last ego trajectory point)
      size_t idx79 = 79 * 4;
      log << "ego traj t=79: " << output.raw_prediction[idx79] << ", "
          << output.raw_prediction[idx79+1] << ", " << output.raw_prediction[idx79+2] << ", "
          << output.raw_prediction[idx79+3] << std::endl;

      log << "turn_indicator_logit: [" << turn_logit[0] << ", "
          << turn_logit[1] << ", " << turn_logit[2] << ", "
          << turn_logit[3] << "]" << std::endl;
      log << std::endl;

      g_frame_counter++;

      // Only log first 100 frames to avoid disk space issues
      if (g_frame_counter >= 100) {
        g_debug_enabled = false;
        log << "=== DEBUG LOGGING STOPPED (100 frames captured) ===" << std::endl;
      }
    }

    output.success = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    output.inference_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    std::cout << "[DiffusionPlanner] Planning completed in "
              << output.inference_time_ms << " ms" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "[DiffusionPlanner] Error during planning: " << e.what() << std::endl;
    output.success = false;
  }

  return output;
}

int DiffusionPlanner::compute_max_feasible_steps(float budget_ms) const
{
  // T_plan(N) = T_enc + (N+1)*T_dit + N*T_sol + T_pre
  // Solve for N: N = floor((budget - T_enc - T_dit - T_pre) / (T_dit + T_sol))
  const float T_enc = 27.5f;   // encoder latency (ms)
  const float T_dit = 2.5f;    // single DiT step (ms, conservative estimate)
  const float T_pre = 2.0f;    // preprocessing
  const float T_sol = 0.01f;   // solver overhead

  float available = budget_ms - T_enc - T_dit - T_pre;
  if (available <= 0) return config_.anytime_min_steps;

  int n = static_cast<int>(available / (T_dit + T_sol));
  return std::max(config_.anytime_min_steps, std::min(n, 20));
}

float DiffusionPlanner::compute_convergence_delta(
  const std::vector<float>& x0_current,
  const std::vector<float>& x0_previous) const
{
  // Compute max displacement of ego trajectory (agent 0) between predictions
  // Ego trajectory: indices [1..80] in sequence dimension, state_dim=4 (x,y,cos,sin)
  // We check position (x,y) only, in normalized coordinates
  float max_delta = 0.0f;
  for (int t = 1; t <= FUTURE_LENGTH; ++t) {
    size_t idx = t * STATE_DIM;  // agent 0, timestep t
    float dx = x0_current[idx] - x0_previous[idx];
    float dy = x0_current[idx + 1] - x0_previous[idx + 1];
    float dist = std::sqrt(dx * dx + dy * dy);
    // Convert to physical meters: multiply by NORM_STD (20m)
    float dist_m = dist * NORM_STD_X;
    if (dist_m > max_delta) max_delta = dist_m;
  }
  return max_delta;
}

}  // namespace autoware::diffusion_planner_onnx_split
