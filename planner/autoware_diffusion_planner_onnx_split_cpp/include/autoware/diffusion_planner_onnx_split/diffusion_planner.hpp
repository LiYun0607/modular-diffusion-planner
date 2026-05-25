// Copyright 2025 TIER IV, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Diffusion Planner v3.0 - ONNX Split Architecture
// Encoder: context_encoder.onnx (runs ONCE, v3: 6 encoder layers)
// DiT:     dit_core_dynamic.onnx (runs N times, dynamic timestep)
// Turn indicator: C++ linear layer (5 classes: NONE/DISABLE/LEFT/RIGHT/KEEP)

#pragma once

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <random>

namespace autoware::diffusion_planner_onnx_split
{

/**
 * @brief Configuration for the diffusion planner
 */
struct DiffusionPlannerConfig
{
  // Model paths (v3.0 split architecture)
  std::string encoder_model_path;          // context_encoder.onnx (runs once)
  std::string dit_model_path;              // dit_core_dynamic.onnx (runs N times)
  std::string turn_indicator_weights_path; // turn_indicator_weights_v3.json

  // DPM-Solver++ settings
  int num_inference_steps = 10;     // Number of denoising steps
  int solver_order = 2;             // 1 = first-order, 2 = second-order (DPM-Solver++)

  // VP noise schedule (continuous)
  float beta_0 = 0.1f;              // beta at t=0
  float beta_1 = 20.0f;             // beta at t=1

  // Inference settings
  bool use_gpu = true;
  int gpu_device_id = 0;

  // Turn indicator manager settings (v3)
  float turn_indicator_keep_offset = -1.25f;    // logit bias for KEEP class
  double turn_indicator_hold_duration = 0.0;    // hold last command for N seconds

  // Anytime planning settings
  bool anytime_enabled = false;           // Enable anytime adaptive planning
  float anytime_budget_ms = 100.0f;       // Total cycle budget in ms
  float anytime_margin_ms = 5.0f;         // Safety margin for deadline guard
  float anytime_convergence_threshold = 0.1f;  // Early stopping threshold (meters)
  int anytime_min_steps = 3;              // Minimum steps (quality floor)
  bool anytime_log_csv = false;           // Log per-cycle data to CSV
  std::string anytime_log_path = "/tmp/anytime_log.csv";
};

/**
 * @brief Input data for the v3 split planner
 * Encoder receives all 16 inputs; DiT receives x + timestep + context_embedding + neighbor_current_mask
 */
struct DiTInput
{
  // Ego vehicle (from autoware)
  std::vector<float> ego_agent_past;        // [31, 4] - x, y, cos, sin
  std::vector<float> ego_current_state;     // [10] - x, y, cos, sin, vx, vy, ax, ay, steer, yaw_rate
  std::vector<float> ego_shape;             // [3] - wheel_base, length, width

  // Neighbor agents
  std::vector<float> neighbor_agents_past;  // [32, 31, 11]
  std::vector<float> static_objects;        // [5, 10]

  // Lanes
  std::vector<float> lanes;                    // [140, 20, 33]
  std::vector<float> lanes_speed_limit;        // [140, 1]
  std::vector<uint8_t> lanes_has_speed_limit;  // [140] as bool

  // Route lanes
  std::vector<float> route_lanes;                    // [25, 20, 33]
  std::vector<float> route_lanes_speed_limit;        // [25, 1]
  std::vector<uint8_t> route_lanes_has_speed_limit;  // [25] as bool

  // Map geometry
  std::vector<float> polygons;              // [10, 40, 2]
  std::vector<float> line_strings;          // [10, 20, 2]

  // Goal and indicators
  std::vector<float> goal_pose;             // [4] - x, y, cos, sin
  std::vector<float> turn_indicators;       // [31]

  // v3: neighbor existence mask (derived from neighbor_agents_past, used by DiT directly)
  // true if neighbor slot is occupied (any non-zero value in last timestep)
  std::vector<bool> neighbor_current_mask;  // [32]

  /// Validate input dimensions
  bool validate() const;

  /// Initialize with zeros (correct shapes)
  static DiTInput create_zeros();
};

/**
 * @brief Output from the planner
 */
struct PlannerOutput
{
  std::vector<float> raw_prediction;       // [33*80*4] - all agents' future trajectories
  std::vector<float> turn_indicator_logit; // [5] - turn indicator logits (filled zeros for truncated model)
  bool success = false;
  float inference_time_ms = 0.0f;

  // Anytime logging info (filled by plan() when anytime_log_csv is enabled)
  float anytime_jitter_ms = 0.0f;
  int anytime_planned_steps = 0;
  int anytime_actual_steps = 0;
  float anytime_denoise_ms = 0.0f;
  float anytime_total_cycle_ms = 0.0f;
  std::string anytime_stop_reason;
  std::vector<float> anytime_per_step_delta;  // delta_i for each step
};

/**
 * @brief VP continuous noise schedule for DPM-Solver++
 *
 * Implements the VP (Variance Preserving) continuous noise schedule:
 *   log_alpha(t) = -0.25 * t^2 * (beta_1 - beta_0) - 0.5 * t * beta_0
 *   alpha(t) = exp(log_alpha(t))
 *   sigma(t) = sqrt(1 - alpha(t)^2)
 */
class VPNoiseSchedule
{
public:
  VPNoiseSchedule(int num_steps, float beta_0 = 0.1f, float beta_1 = 20.0f);

  /// Get continuous timestep for step index (logSNR spacing)
  float get_timestep(int step_index) const;

  /// Get alpha at continuous timestep t
  float get_alpha(float t) const;

  /// Get sigma at continuous timestep t
  float get_sigma(float t) const;

  /// Get log(alpha) at t (VP schedule formula)
  float log_alpha(float t) const;

  /// Get lambda (log-SNR) at t
  float lambda(float t) const;

  /// Inverse: get t from lambda
  float inverse_lambda(float lambda_val) const;

  int num_steps() const { return num_steps_; }
  const std::vector<float>& timesteps() const { return timesteps_; }

private:
  int num_steps_;
  float beta_0_;
  float beta_1_;
  std::vector<float> timesteps_;  // Continuous timesteps [t_0, t_1, ..., t_N]
};

/**
 * @brief Main Diffusion Planner class using GraphSurgeon split architecture
 *
 * This class implements:
 * - Context encoder (runs ONCE per planning cycle)
 * - DiT core (runs N times in DPM-Solver loop)
 * - Turn indicator computation in C++
 * - DPM-Solver++ denoising loop
 * - VP continuous noise schedule
 */
class DiffusionPlanner
{
public:
  explicit DiffusionPlanner(const DiffusionPlannerConfig& config);
  ~DiffusionPlanner();

  // Prevent copying
  DiffusionPlanner(const DiffusionPlanner&) = delete;
  DiffusionPlanner& operator=(const DiffusionPlanner&) = delete;

  /**
   * @brief Run the full planning pipeline
   * @param input DiT input data (context + map)
   * @return Planning output with trajectory
   */
  PlannerOutput plan(const DiTInput& input);

  /**
   * @brief Run single DiT step (for debugging)
   * @param noisy_trajectory Current trajectory [33, 81, 4]
   * @param timestep Continuous timestep (0=clean, 1=noise)
   * @return Model prediction (x0)
   */
  std::vector<float> run_dit_step(
    const std::vector<float>& noisy_trajectory,
    float timestep);

  /// Check if planner is ready
  bool is_ready() const { return initialized_; }

  /// Access configuration (for node-level CSV logging)
  const DiffusionPlannerConfig& get_config() const { return config_; }

private:
  // Initialize ONNX Runtime sessions
  void init_sessions();

  // Load turn indicator weights from JSON
  void load_turn_indicator_weights();

  // Run encoder once and cache context_embedding
  void run_encoder(const DiTInput& input);

  // Create input tensors for encoder
  std::vector<Ort::Value> create_encoder_inputs(const DiTInput& input);

  // Create input tensors for DiT core
  std::vector<Ort::Value> create_dit_inputs(
    const std::vector<float>& sampled_trajectory,
    float timestep);

public:
  // Apply TurnIndicatorManager logic (KEEP hysteresis + hold_duration) -> final command
  uint8_t evaluate_turn_indicator(
    const std::vector<float>& logit, int64_t stamp_ns, uint8_t prev_report);

private:
  // Compute raw turn indicator logits [5] from final trajectory
  std::vector<float> compute_turn_indicator(const std::vector<float>& x0);

  // Sample initial noise trajectory
  std::vector<float> sample_initial_noise();

  // DPM-Solver++ first-order update
  void dpm_solver_first_order_update(
    std::vector<float>& x_t,
    const std::vector<float>& model_output,
    float t_current,
    float t_next);

  // Apply initial state constraint (set t=0 to current_states)
  void apply_initial_state_constraint(
    std::vector<float>& x_t,
    const std::vector<float>& current_states);

  // DPM-Solver++ second-order update (multistep)
  void dpm_solver_second_order_update(
    std::vector<float>& x_t,
    const std::vector<float>& model_output_0,
    const std::vector<float>& model_output_1,
    float t_prev,
    float t_current,
    float t_next);

  // Denormalize trajectory output
  void denormalize_trajectory(std::vector<float>& trajectory);

private:
  DiffusionPlannerConfig config_;

  // ONNX Runtime
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> encoder_session_;  // context_encoder.onnx (v3)
  std::unique_ptr<Ort::Session> dit_session_;      // dit_core_dynamic.onnx (v3)
  Ort::MemoryInfo memory_info_{nullptr};

  // Input/Output names
  std::vector<const char*> encoder_input_names_;
  std::vector<const char*> encoder_output_names_;
  std::vector<const char*> dit_input_names_;
  std::vector<const char*> dit_output_names_;

  // Cached encoder output
  std::vector<float> cached_context_embedding_;  // [1, 226, 256] (v3: batch-first)
  std::vector<float> cached_encoding_pooled_;   // [256] mean pooled for turn indicator

  // Cached DiTInput data for dit_core (v3: only neighbor_current_mask needed in loop)
  std::vector<bool> cached_neighbor_current_mask_;  // [32]

  // Turn indicator weights (Linear: 272 -> 5, v3)
  std::vector<float> turn_indicator_weight_;  // [5, 272]
  std::vector<float> turn_indicator_bias_;    // [5]

  // TurnIndicatorManager state (KEEP class hysteresis)
  int64_t last_non_keep_command_ = 1;  // default: DISABLE
  int64_t last_non_keep_stamp_ns_ = 0;
  float turn_indicator_keep_offset_ = -1.25f;
  int64_t turn_indicator_hold_duration_ns_ = 0;

  // Noise schedule
  std::unique_ptr<VPNoiseSchedule> noise_schedule_;

  // Random number generator
  std::mt19937 rng_;
  std::normal_distribution<float> normal_dist_;

  // Anytime planning internals
  int compute_max_feasible_steps(float budget_ms) const;
  float compute_convergence_delta(
    const std::vector<float>& x0_current,
    const std::vector<float>& x0_previous) const;
  std::ofstream anytime_log_file_;
  bool anytime_log_header_written_ = false;

  // State
  bool initialized_ = false;

  // Constants (v3.0)
  static constexpr int BATCH_SIZE = 1;
  static constexpr int NUM_SAMPLES = 33;         // ego + 32 neighbors
  static constexpr int NUM_NEIGHBORS = 32;
  static constexpr int SEQUENCE_LENGTH = 81;     // 1 current + 80 future
  static constexpr int STATE_DIM = 4;            // x, y, cos, sin
  static constexpr int FUTURE_LENGTH = 80;       // 80 future timesteps
  static constexpr int CONTEXT_SEQ_LEN = 226;    // context_embedding sequence length
  static constexpr int EMBED_DIM = 256;          // embedding dimension
  static constexpr int DIT_X_DIM = 324;          // NUM_SAMPLES * STATE_DIM = 81*4 (flattened)
  static constexpr int TURN_INDICATOR_CLASSES = 5; // NONE/DISABLE/LEFT/RIGHT/KEEP
  static constexpr int TURN_INDICATOR_KEEP = 4;

  // Normalization constants (from training, same as v2)
  static constexpr float NORM_MEAN_X = 10.0f;
  static constexpr float NORM_MEAN_Y = 0.0f;
  static constexpr float NORM_STD_X = 20.0f;
  static constexpr float NORM_STD_Y = 20.0f;
};

}  // namespace autoware::diffusion_planner_onnx_split
