// Copyright 2025 TIER IV, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Diffusion Planner Node with dit_single_step.onnx - Pure C++ Version
// Uses preprocessing/postprocessing utilities from autoware_diffusion_planner

#include "autoware/diffusion_planner_onnx_split/diffusion_planner.hpp"

// Preprocessing utilities from original diffusion planner
#include <autoware/diffusion_planner/conversion/ego.hpp>
#include <autoware/diffusion_planner/conversion/agent.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/diffusion_planner/postprocessing/postprocessing_utils.hpp>
#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>
#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <autoware/diffusion_planner/preprocessing/traffic_signals.hpp>
#include <autoware/diffusion_planner/utils/arg_reader.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>
#include <autoware/lanelet2_utils/conversion.hpp>
#include <autoware/vehicle_info_utils/vehicle_info.hpp>
#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <autoware_internal_planning_msgs/msg/candidate_trajectories.hpp>
#include <autoware_internal_planning_msgs/msg/candidate_trajectory.hpp>
#include <autoware_internal_planning_msgs/msg/generator_info.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>
#include <std_msgs/msg/string.hpp>
#include <autoware_utils_uuid/uuid_helper.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_command.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <autoware_utils_diagnostics/diagnostics_interface.hpp>

#include <Eigen/Dense>
#include <lanelet2_core/LaneletMap.h>

#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::diffusion_planner_onnx_split
{

// Import from original diffusion planner
using namespace autoware::diffusion_planner;

using nav_msgs::msg::Odometry;
using geometry_msgs::msg::AccelWithCovarianceStamped;
using geometry_msgs::msg::Pose;
using autoware_internal_planning_msgs::msg::CandidateTrajectories;
using autoware_planning_msgs::msg::Trajectory;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_perception_msgs::msg::TrackedObjects;
using autoware_perception_msgs::msg::TrafficLightGroupArray;
using autoware_map_msgs::msg::LaneletMapBin;
using autoware_planning_msgs::msg::LaneletRoute;
using autoware_vehicle_msgs::msg::TurnIndicatorsCommand;
using autoware_vehicle_msgs::msg::TurnIndicatorsReport;
using visualization_msgs::msg::MarkerArray;
using autoware::vehicle_info_utils::VehicleInfo;
using preprocess::TrafficSignalStamped;
using unique_identifier_msgs::msg::UUID;
using InputDataMap = std::unordered_map<std::string, std::vector<float>>;

class DiffusionPlannerNode : public rclcpp::Node
{
public:
  explicit DiffusionPlannerNode(const rclcpp::NodeOptions & options)
  : Node("diffusion_planner", options),
    generator_uuid_(autoware_utils_uuid::generate_uuid())
  {
    RCLCPP_INFO(get_logger(), "============================================");
    RCLCPP_INFO(get_logger(), "  Diffusion Planner (GraphSurgeon Split)");
    RCLCPP_INFO(get_logger(), "  - Encoder: runs ONCE per cycle");
    RCLCPP_INFO(get_logger(), "  - DiT Core: runs N times (DPM-Solver++)");
    RCLCPP_INFO(get_logger(), "  - Turn Indicator: C++ computation");
    RCLCPP_INFO(get_logger(), "============================================");

    // Declare parameters
    declare_parameters();

    // Get vehicle info
    vehicle_info_ = autoware::vehicle_info_utils::VehicleInfoUtils(*this).getVehicleInfo();
    RCLCPP_INFO(get_logger(), "Vehicle wheel_base: %.2f m", vehicle_info_.wheel_base_m);

    // Load normalization stats from JSON
    {
      std::string args_path = expand_path(this->get_parameter("args_path").as_string());
      normalization_map_ = utils::load_normalization_stats(args_path);
      RCLCPP_INFO(get_logger(), "Loaded normalization stats from: %s", args_path.c_str());
    }

    // Read postprocessing parameters
    velocity_smoothing_window_ = this->get_parameter("velocity_smoothing_window").as_int();
    stopping_threshold_ = this->get_parameter("stopping_threshold").as_double();
    predict_neighbor_trajectory_ = this->get_parameter("predict_neighbor_trajectory").as_bool();

    // Initialize the planner
    init_planner();

    // Publishers
    pub_trajectory_ = create_publisher<Trajectory>("~/output/trajectory", 1);
    pub_trajectories_ = create_publisher<CandidateTrajectories>("~/output/trajectories", 1);
    pub_objects_ = create_publisher<PredictedObjects>("~/output/predicted_objects", 1);
    pub_turn_indicators_ = create_publisher<TurnIndicatorsCommand>("~/output/turn_indicators", 1);
    pub_debug_marker_ = create_publisher<MarkerArray>("~/debug/marker", 10);

    // Diagnostics interface (required for auto mode!)
    diagnostics_inference_ = std::make_unique<autoware_utils_diagnostics::DiagnosticsInterface>(
      this, "inference_status");

    // Subscribers - using relative topic names for launch file remapping
    sub_odometry_ = create_subscription<Odometry>(
      "~/input/odometry", 1,
      std::bind(&DiffusionPlannerNode::on_odometry, this, std::placeholders::_1));

    sub_acceleration_ = create_subscription<AccelWithCovarianceStamped>(
      "~/input/acceleration", 1,
      std::bind(&DiffusionPlannerNode::on_acceleration, this, std::placeholders::_1));

    sub_tracked_objects_ = create_subscription<TrackedObjects>(
      "~/input/tracked_objects", 1,
      std::bind(&DiffusionPlannerNode::on_tracked_objects, this, std::placeholders::_1));

    sub_traffic_signals_ = create_subscription<TrafficLightGroupArray>(
      "~/input/traffic_signals", rclcpp::QoS{10},
      std::bind(&DiffusionPlannerNode::on_traffic_signals, this, std::placeholders::_1));

    sub_route_ = create_subscription<LaneletRoute>(
      "~/input/route", rclcpp::QoS{1}.transient_local(),
      std::bind(&DiffusionPlannerNode::on_route, this, std::placeholders::_1));

    sub_map_ = create_subscription<LaneletMapBin>(
      "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
      std::bind(&DiffusionPlannerNode::on_map, this, std::placeholders::_1));

    sub_turn_indicators_ = create_subscription<TurnIndicatorsReport>(
      "~/input/turn_indicators", 1,
      std::bind(&DiffusionPlannerNode::on_turn_indicators, this, std::placeholders::_1));

    // Timer for planning loop
    const double planning_frequency = this->get_parameter("planning_frequency_hz").as_double();
    const auto period = std::chrono::duration<double>(1.0 / planning_frequency);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&DiffusionPlannerNode::on_timer, this));

    RCLCPP_INFO(get_logger(), "Node initialized. Planning at %.1f Hz", planning_frequency);
  }

  ~DiffusionPlannerNode() override = default;

private:
  static std::string expand_path(const std::string& path)
  {
    if (path.empty() || path[0] != '~') {
      return path;
    }
    const char* home = std::getenv("HOME");
    return home ? std::string(home) + path.substr(1) : path;
  }

  void declare_parameters()
  {
    // v3.0 split model paths
    this->declare_parameter<std::string>("encoder_model_path",
      "~/autoware_data/diffusion_planner/v3_from_pth/context_encoder.onnx");
    this->declare_parameter<std::string>("dit_model_path",
      "~/autoware_data/diffusion_planner/v3_from_pth/dit_core_dynamic.onnx");
    this->declare_parameter<std::string>("turn_indicator_weights_path",
      "~/autoware_data/diffusion_planner/v3_from_pth/turn_indicator_weights_v3.json");

    // Normalization stats JSON path (v3.0)
    this->declare_parameter<std::string>("args_path",
      "~/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json");

    // DPM-Solver++ settings
    this->declare_parameter<int>("num_inference_steps", 10);
    this->declare_parameter<int>("solver_order", 2);  // 1=first-order, 2=second-order

    // VP noise schedule parameters
    this->declare_parameter<double>("beta_0", 0.1);
    this->declare_parameter<double>("beta_1", 20.0);

    // GPU settings
    this->declare_parameter<bool>("use_gpu", true);
    this->declare_parameter<int>("gpu_device_id", 0);

    // Planning settings
    this->declare_parameter<double>("planning_frequency_hz", 10.0);
    this->declare_parameter<double>("traffic_light_group_msg_timeout_seconds", 0.2);
    this->declare_parameter<bool>("ignore_neighbors", false);
    this->declare_parameter<bool>("ignore_unknown_neighbors", false);
    this->declare_parameter<double>("temperature", 0.0);

    // Postprocessing settings
    this->declare_parameter<int64_t>("velocity_smoothing_window", 8);
    this->declare_parameter<double>("stopping_threshold", 0.3);
    this->declare_parameter<bool>("predict_neighbor_trajectory", true);

    // Turn indicator manager settings (v3)
    this->declare_parameter<double>("turn_indicator_keep_offset", -1.25);
    this->declare_parameter<double>("turn_indicator_hold_duration", 0.0);

    // Anytime planning settings
    this->declare_parameter<bool>("anytime_enabled", false);
    this->declare_parameter<double>("anytime_budget_ms", 100.0);
    this->declare_parameter<double>("anytime_margin_ms", 5.0);
    this->declare_parameter<double>("anytime_convergence_threshold", 0.1);
    this->declare_parameter<int>("anytime_min_steps", 3);
    this->declare_parameter<bool>("anytime_log_csv", false);
    this->declare_parameter<std::string>("anytime_log_path", "/tmp/anytime_log.csv");
  }

  void init_planner()
  {
    DiffusionPlannerConfig config;

    config.encoder_model_path = expand_path(
      this->get_parameter("encoder_model_path").as_string());
    config.dit_model_path = expand_path(
      this->get_parameter("dit_model_path").as_string());
    config.turn_indicator_weights_path = expand_path(
      this->get_parameter("turn_indicator_weights_path").as_string());
    config.num_inference_steps = this->get_parameter("num_inference_steps").as_int();
    config.solver_order = this->get_parameter("solver_order").as_int();
    config.beta_0 = static_cast<float>(this->get_parameter("beta_0").as_double());
    config.beta_1 = static_cast<float>(this->get_parameter("beta_1").as_double());
    config.use_gpu = this->get_parameter("use_gpu").as_bool();
    config.gpu_device_id = this->get_parameter("gpu_device_id").as_int();

    // Anytime settings
    config.turn_indicator_keep_offset =
      static_cast<float>(this->get_parameter("turn_indicator_keep_offset").as_double());
    config.turn_indicator_hold_duration =
      this->get_parameter("turn_indicator_hold_duration").as_double();

    config.anytime_enabled = this->get_parameter("anytime_enabled").as_bool();
    config.anytime_budget_ms = static_cast<float>(this->get_parameter("anytime_budget_ms").as_double());
    config.anytime_margin_ms = static_cast<float>(this->get_parameter("anytime_margin_ms").as_double());
    config.anytime_convergence_threshold = static_cast<float>(this->get_parameter("anytime_convergence_threshold").as_double());
    config.anytime_min_steps = this->get_parameter("anytime_min_steps").as_int();
    config.anytime_log_csv = this->get_parameter("anytime_log_csv").as_bool();
    config.anytime_log_path = this->get_parameter("anytime_log_path").as_string();

    RCLCPP_INFO(get_logger(), "Encoder model: %s", config.encoder_model_path.c_str());
    RCLCPP_INFO(get_logger(), "DiT model: %s", config.dit_model_path.c_str());
    RCLCPP_INFO(get_logger(), "Turn indicator weights: %s", config.turn_indicator_weights_path.c_str());
    RCLCPP_INFO(get_logger(), "Inference steps: %d", config.num_inference_steps);
    RCLCPP_INFO(get_logger(), "Solver order: %d (%s)", config.solver_order,
                config.solver_order == 1 ? "first-order" : "second-order DPM-Solver++");
    RCLCPP_INFO(get_logger(), "VP schedule: beta_0=%.2f, beta_1=%.2f", config.beta_0, config.beta_1);
    if (config.anytime_enabled) {
      RCLCPP_INFO(get_logger(), "Anytime planning ENABLED: budget=%.0fms, threshold=%.2fm, min_steps=%d",
                  config.anytime_budget_ms, config.anytime_convergence_threshold, config.anytime_min_steps);
    }

    try {
      planner_ = std::make_unique<DiffusionPlanner>(config);
      RCLCPP_INFO(get_logger(), "Planner initialized successfully!");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize planner: %s", e.what());
      throw;
    }
  }

  // --- Subscription callbacks ---

  void on_odometry(const Odometry::SharedPtr msg) { current_odometry_ = msg; }
  void on_acceleration(const AccelWithCovarianceStamped::SharedPtr msg) { current_acceleration_ = msg; }
  void on_tracked_objects(const TrackedObjects::SharedPtr msg) { current_objects_ = msg; }
  void on_traffic_signals(const TrafficLightGroupArray::SharedPtr msg) { current_traffic_signals_ = msg; }
  void on_turn_indicators(const TurnIndicatorsReport::SharedPtr msg) { current_turn_indicators_ = msg; }

  void on_route(const LaneletRoute::SharedPtr msg)
  {
    current_route_ = msg;
    RCLCPP_INFO(get_logger(), "Route received");
  }

  void on_map(const LaneletMapBin::SharedPtr msg)
  {
    current_map_ = msg;
    try {
      auto lanelet_map_ptr = std::const_pointer_cast<lanelet::LaneletMap>(
          autoware::experimental::lanelet2_utils::from_autoware_map_msgs(*msg));
      lane_segment_context_ = std::make_unique<preprocess::LaneSegmentContext>(lanelet_map_ptr);
      is_map_loaded_ = true;
      RCLCPP_INFO(get_logger(), "Map loaded and LaneSegmentContext created");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to process map: %s", e.what());
    }
  }

  // --- Normalization helpers ---

  InputDataMap dit_input_to_input_data_map(const DiTInput& input) const
  {
    InputDataMap map;
    map["ego_agent_past"] = input.ego_agent_past;
    map["ego_current_state"] = input.ego_current_state;
    map["ego_shape"] = input.ego_shape;
    map["neighbor_agents_past"] = input.neighbor_agents_past;
    map["static_objects"] = input.static_objects;
    map["lanes"] = input.lanes;
    map["lanes_speed_limit"] = input.lanes_speed_limit;
    map["route_lanes"] = input.route_lanes;
    map["route_lanes_speed_limit"] = input.route_lanes_speed_limit;
    map["polygons"] = input.polygons;
    map["line_strings"] = input.line_strings;
    map["goal_pose"] = input.goal_pose;
    map["turn_indicators"] = input.turn_indicators;
    // Note: sampled_trajectories NOT included - handled internally by DPM-Solver++
    return map;
  }

  void apply_normalized_data_to_dit_input(DiTInput& input, const InputDataMap& map) const
  {
    // Only update fields that are actually normalized
    // (normalize_input_data skips ego_shape, sampled_trajectories, turn_indicators)
    input.ego_agent_past = map.at("ego_agent_past");
    input.ego_current_state = map.at("ego_current_state");
    input.neighbor_agents_past = map.at("neighbor_agents_past");
    input.static_objects = map.at("static_objects");
    input.lanes = map.at("lanes");
    input.lanes_speed_limit = map.at("lanes_speed_limit");
    input.route_lanes = map.at("route_lanes");
    input.route_lanes_speed_limit = map.at("route_lanes_speed_limit");
    input.polygons = map.at("polygons");
    input.line_strings = map.at("line_strings");
    input.goal_pose = map.at("goal_pose");
  }

  // --- Preprocessing ---

  std::optional<DiTInput> create_dit_input()
  {
    if (!current_odometry_ || !current_acceleration_ || !current_route_ ||
        !current_turn_indicators_ || !lane_segment_context_) {
      return std::nullopt;
    }

    const auto& ego_kinematic_state = *current_odometry_;
    const auto& ego_acceleration = *current_acceleration_;
    const Pose& pose_base_link = ego_kinematic_state.pose.pose;

    const Eigen::Matrix4d ego_to_map_transform = utils::pose_to_matrix4f(pose_base_link);
    const Eigen::Matrix4d map_to_ego_transform = utils::inverse(ego_to_map_transform);

    // Update ego history (store Pose for create_ego_agent_past)
    ego_history_odom_.push_back(pose_base_link);
    if (ego_history_odom_.size() > static_cast<size_t>(EGO_HISTORY_SHAPE[1])) {
      ego_history_odom_.pop_front();
    }

    // Update turn indicators history
    turn_indicators_history_.push_back(*current_turn_indicators_);
    if (turn_indicators_history_.size() > static_cast<size_t>(INPUT_T + 1)) {
      turn_indicators_history_.pop_front();
    }

    // Update neighbor agent data
    TrackedObjects objects;
    if (current_objects_) {
      objects = this->get_parameter("ignore_neighbors").as_bool()
        ? TrackedObjects() : *current_objects_;
    }

    const bool ignore_unknown = this->get_parameter("ignore_unknown_neighbors").as_bool();
    if (!agent_data_) {
      agent_data_ = AgentData(objects, NEIGHBOR_SHAPE[1], NEIGHBOR_SHAPE[2], ignore_unknown);
    } else {
      agent_data_->update_histories(objects, ignore_unknown);
    }
    auto ego_centric_neighbor_agent_data = agent_data_.value();
    ego_centric_neighbor_agent_data.apply_transform(map_to_ego_transform);
    ego_centric_neighbor_agent_data.trim_to_k_closest_agents();

    // Save for postprocessing (neighbor predictions)
    last_ego_centric_agent_data_ = ego_centric_neighbor_agent_data;

    // Update traffic light map
    if (current_traffic_signals_) {
      const double timeout_s = this->get_parameter("traffic_light_group_msg_timeout_seconds").as_double();
      preprocess::process_traffic_signals(
        current_traffic_signals_, traffic_light_id_map_, this->now(), timeout_s, true);
    }

    const auto center_x = pose_base_link.position.x;
    const auto center_y = pose_base_link.position.y;
    const auto center_z = pose_base_link.position.z;

    DiTInput input;

    // --- Ego data ---
    input.ego_agent_past = preprocess::create_ego_agent_past(
      ego_history_odom_, EGO_HISTORY_SHAPE[1], map_to_ego_transform);

    input.ego_current_state = EgoState(
      ego_kinematic_state, ego_acceleration,
      static_cast<float>(vehicle_info_.wheel_base_m)).as_array();

    const float wheel_base = static_cast<float>(vehicle_info_.wheel_base_m);
    const float vehicle_length = static_cast<float>(
      vehicle_info_.front_overhang_m + vehicle_info_.wheel_base_m + vehicle_info_.rear_overhang_m);
    const float vehicle_width = static_cast<float>(
      vehicle_info_.left_overhang_m + vehicle_info_.wheel_tread_m + vehicle_info_.right_overhang_m);
    input.ego_shape = {wheel_base, vehicle_length, vehicle_width};

    // --- Neighbor data ---
    input.neighbor_agents_past = ego_centric_neighbor_agent_data.as_vector();
    input.static_objects.resize(5 * 10, 0.0f);

    // v3: build neighbor_current_mask - true if neighbor slot has any non-zero data
    // neighbor_agents_past shape: [32, 31, 11]; check last timestep (index 30) per neighbor
    {
      constexpr int N_T = 31, N_D = 11;
      input.neighbor_current_mask.resize(32, false);
      for (int n = 0; n < 32; ++n) {
        const size_t base = static_cast<size_t>(n * N_T + 30) * N_D;
        for (int d = 0; d < N_D; ++d) {
          if (std::abs(input.neighbor_agents_past[base + d]) >
              std::numeric_limits<float>::epsilon()) {
            input.neighbor_current_mask[n] = true;
            break;
          }
        }
      }
    }

    // --- Lane data ---
    {
      const auto segment_indices = lane_segment_context_->select_lane_segment_indices(
        map_to_ego_transform, center_x, center_y, NUM_SEGMENTS_IN_LANE);
      const auto [lanes, lanes_speed_limit] = lane_segment_context_->create_tensor_data_from_indices(
        map_to_ego_transform, traffic_light_id_map_, segment_indices, NUM_SEGMENTS_IN_LANE);
      input.lanes = lanes;
      input.lanes_speed_limit = lanes_speed_limit;
      input.lanes_has_speed_limit.resize(NUM_SEGMENTS_IN_LANE);
      for (int i = 0; i < NUM_SEGMENTS_IN_LANE; ++i) {
        input.lanes_has_speed_limit[i] = (lanes_speed_limit[i] != 0.0f) ? 1 : 0;
      }
    }

    // --- Route lane data ---
    {
      const auto segment_indices = lane_segment_context_->select_route_segment_indices(
        *current_route_, center_x, center_y, static_cast<int64_t>(NUM_SEGMENTS_IN_ROUTE));
      const auto [route_lanes, route_speed] = lane_segment_context_->create_tensor_data_from_indices(
        map_to_ego_transform, traffic_light_id_map_, segment_indices, NUM_SEGMENTS_IN_ROUTE);
      input.route_lanes = route_lanes;
      input.route_lanes_speed_limit = route_speed;
      input.route_lanes_has_speed_limit.resize(NUM_SEGMENTS_IN_ROUTE);
      for (int i = 0; i < NUM_SEGMENTS_IN_ROUTE; ++i) {
        input.route_lanes_has_speed_limit[i] = (route_speed[i] != 0.0f) ? 1 : 0;
      }
    }

    // --- Polygons and line strings ---
    // NOTE: create_polygon_tensor / create_line_string_tensor are not available in the
    // installed version of autoware_diffusion_planner. Fill with zeros as placeholder.
    input.polygons.assign(10 * 40 * 2, 0.0f);       // [NUM_POLYGONS, POINTS_PER_POLYGON, 2]
    input.line_strings.assign(10 * 20 * 2, 0.0f);   // [NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2]

    // --- Goal pose ---
    {
      const auto& goal_pose = current_route_->goal_pose;
      const Eigen::Matrix4d goal_map = utils::pose_to_matrix4f(goal_pose);
      const Eigen::Matrix4d goal_ego = map_to_ego_transform * goal_map;
      const float x = goal_ego(0, 3);
      const float y = goal_ego(1, 3);
      const auto [cos_yaw, sin_yaw] = utils::rotation_matrix_to_cos_sin(
        goal_ego.block<3, 3>(0, 0));
      input.goal_pose = {x, y, cos_yaw, sin_yaw};
    }

    // --- Turn indicators ---
    {
      input.turn_indicators.resize(INPUT_T + 1, 0.0f);
      for (int64_t t = 0; t < INPUT_T + 1; ++t) {
        const int64_t index = std::max(
          static_cast<int64_t>(turn_indicators_history_.size()) - 1 - t,
          static_cast<int64_t>(0));
        input.turn_indicators[INPUT_T - t] = turn_indicators_history_[index].report;
      }
    }

    return input;
  }

  // --- Main timer callback ---

  void on_timer()
  {
    // Clear diagnostics at the start of each cycle
    diagnostics_inference_->clear();

    if (!is_map_loaded_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for map...");
      diagnostics_inference_->update_level_and_message(
        diagnostic_msgs::msg::DiagnosticStatus::WARN, "Map data not loaded");
      diagnostics_inference_->publish(this->now());
      return;
    }

    if (!current_odometry_ || !current_route_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Waiting for input data (odometry: %s, route: %s)",
        current_odometry_ ? "OK" : "MISSING",
        current_route_ ? "OK" : "MISSING");
      diagnostics_inference_->update_level_and_message(
        diagnostic_msgs::msg::DiagnosticStatus::WARN, "Waiting for input data");
      diagnostics_inference_->publish(this->now());
      return;
    }

    if (!planner_ || !planner_->is_ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Planner not ready");
      diagnostics_inference_->update_level_and_message(
        diagnostic_msgs::msg::DiagnosticStatus::WARN, "Planner not ready");
      diagnostics_inference_->publish(this->now());
      return;
    }

    // Create DiTInput from ROS messages
    auto dit_input_opt = create_dit_input();
    if (!dit_input_opt) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Failed to create DiT input - missing data (accel: %s, turn: %s, lane_ctx: %s)",
        current_acceleration_ ? "OK" : "MISSING",
        current_turn_indicators_ ? "OK" : "MISSING",
        lane_segment_context_ ? "OK" : "MISSING");
      diagnostics_inference_->update_level_and_message(
        diagnostic_msgs::msg::DiagnosticStatus::WARN, "Missing input data for planning");
      diagnostics_inference_->publish(this->now());
      return;
    }

    DiTInput dit_input = std::move(*dit_input_opt);

    // Normalize inputs (critical for correct model output)
    InputDataMap input_data_map = dit_input_to_input_data_map(dit_input);
    preprocess::normalize_input_data(input_data_map, normalization_map_);
    if (!utils::check_input_map(input_data_map)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Input data contains invalid values after normalization");
      diagnostics_inference_->update_level_and_message(
        diagnostic_msgs::msg::DiagnosticStatus::WARN, "Invalid input data");
      diagnostics_inference_->publish(this->now());
      return;
    }
    apply_normalized_data_to_dit_input(dit_input, input_data_map);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "[DiT Single-Step] Planning cycle with real preprocessing");

    // Run planning
    auto output = planner_->plan(dit_input);

    if (!output.success) {
      RCLCPP_WARN(get_logger(), "Planning failed!");
      diagnostics_inference_->update_level_and_message(
        diagnostic_msgs::msg::DiagnosticStatus::ERROR, "Planning inference failed");
      diagnostics_inference_->publish(this->now());
      return;
    }

    // --- Anytime CSV logging with ego pose ---
    if (planner_->get_config().anytime_log_csv && current_odometry_) {
      const auto& pose = current_odometry_->pose.pose;
      const double vx = current_odometry_->twist.twist.linear.x;
      const double yaw = std::atan2(
        2.0 * (pose.orientation.w * pose.orientation.z + pose.orientation.x * pose.orientation.y),
        1.0 - 2.0 * (pose.orientation.y * pose.orientation.y + pose.orientation.z * pose.orientation.z));

      if (!anytime_csv_file_.is_open()) {
        anytime_csv_file_.open(planner_->get_config().anytime_log_path, std::ios::app);
      }
      if (!anytime_csv_header_written_) {
        anytime_csv_file_ << "jitter_ms,planned_steps,actual_steps,denoise_ms,total_cycle_ms,"
                          << "stop_reason,ego_x,ego_y,ego_yaw,ego_vx";
        // per-step delta columns
        for (int s = 0; s < planner_->get_config().num_inference_steps; ++s) {
          anytime_csv_file_ << ",delta_" << s;
        }
        anytime_csv_file_ << std::endl;
        anytime_csv_header_written_ = true;
      }
      anytime_csv_file_ << output.anytime_jitter_ms << ","
                        << output.anytime_planned_steps << ","
                        << output.anytime_actual_steps << ","
                        << output.anytime_denoise_ms << ","
                        << output.anytime_total_cycle_ms << ","
                        << output.anytime_stop_reason << ","
                        << std::fixed << std::setprecision(2)
                        << pose.position.x << ","
                        << pose.position.y << ","
                        << std::setprecision(4) << yaw << ","
                        << std::setprecision(2) << vx;
      // per-step deltas (pad with empty if fewer steps)
      for (int s = 0; s < planner_->get_config().num_inference_steps; ++s) {
        if (s < static_cast<int>(output.anytime_per_step_delta.size())) {
          anytime_csv_file_ << "," << std::setprecision(4) << output.anytime_per_step_delta[s];
        } else {
          anytime_csv_file_ << ",";
        }
      }
      anytime_csv_file_ << std::endl;
      anytime_csv_file_.flush();
    }

    // --- Postprocessing using original utilities ---
    const Pose& pose_base_link = current_odometry_->pose.pose;
    const Eigen::Matrix4d ego_to_map_transform = utils::pose_to_matrix4f(pose_base_link);
    const int64_t batch_idx = 0;

    // parse_predictions returns poses in ego frame
    const auto agent_poses = postprocess::parse_predictions(output.raw_prediction);

    // Determine if force-stop should be enabled
    const bool enable_force_stop =
      current_odometry_->twist.twist.linear.x > std::numeric_limits<double>::epsilon();

    const Trajectory trajectory = postprocess::create_ego_trajectory(
      agent_poses, this->now(), ego_to_map_transform, batch_idx,
      velocity_smoothing_window_, enable_force_stop, stopping_threshold_);
    pub_trajectory_->publish(trajectory);

    // Publish candidate trajectories (required by trajectory_optimizer)
    CandidateTrajectories candidate_trajectories;

    // Build candidate trajectory message using the builder pattern
    const auto candidate_trajectory = autoware_internal_planning_msgs::build<
                                        autoware_internal_planning_msgs::msg::CandidateTrajectory>()
                                        .header(trajectory.header)
                                        .generator_id(generator_uuid_)
                                        .points(trajectory.points);

    std_msgs::msg::String generator_name_msg;
    generator_name_msg.data = "DiffusionPlannerOnnxSplit";

    const auto generator_info =
      autoware_internal_planning_msgs::build<autoware_internal_planning_msgs::msg::GeneratorInfo>()
        .generator_id(generator_uuid_)
        .generator_name(generator_name_msg);

    candidate_trajectories.candidate_trajectories.push_back(candidate_trajectory);
    candidate_trajectories.generator_info.push_back(generator_info);
    pub_trajectories_->publish(candidate_trajectories);

    // v3: TurnIndicatorManager (KEEP hysteresis + hold_duration)
    const int64_t stamp_ns = this->now().nanoseconds();
    const uint8_t prev_report = current_turn_indicators_
      ? static_cast<uint8_t>(current_turn_indicators_->report) : 1u;
    TurnIndicatorsCommand turn_cmd;
    turn_cmd.stamp = this->now();
    turn_cmd.command = planner_->evaluate_turn_indicator(
      output.turn_indicator_logit, stamp_ns, prev_report);
    pub_turn_indicators_->publish(turn_cmd);

    // Publish neighbor predictions
    if (predict_neighbor_trajectory_ && last_ego_centric_agent_data_) {
      const auto predicted_objects = postprocess::create_predicted_objects(
        agent_poses, *last_ego_centric_agent_data_, this->now(), ego_to_map_transform, batch_idx);
      pub_objects_->publish(predicted_objects);
    }

    // Publish diagnostics - planning successful!
    diagnostics_inference_->update_level_and_message(
      diagnostic_msgs::msg::DiagnosticStatus::OK, "Planning OK");
    diagnostics_inference_->add_key_value("inference_time_ms", output.inference_time_ms);
    diagnostics_inference_->add_key_value("trajectory_points", static_cast<int>(trajectory.points.size()));
    diagnostics_inference_->publish(this->now());

    RCLCPP_DEBUG(get_logger(), "Published trajectory with %zu points (%.1f ms)",
      trajectory.points.size(), output.inference_time_ms);
  }

private:
  // Planner
  std::unique_ptr<DiffusionPlanner> planner_;
  UUID generator_uuid_;  // Unique identifier for trajectory generator

  // Publishers
  rclcpp::Publisher<Trajectory>::SharedPtr pub_trajectory_;
  rclcpp::Publisher<CandidateTrajectories>::SharedPtr pub_trajectories_;
  rclcpp::Publisher<PredictedObjects>::SharedPtr pub_objects_;
  rclcpp::Publisher<TurnIndicatorsCommand>::SharedPtr pub_turn_indicators_;
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_debug_marker_;

  // Diagnostics
  std::unique_ptr<autoware_utils_diagnostics::DiagnosticsInterface> diagnostics_inference_;

  // Subscribers
  rclcpp::Subscription<Odometry>::SharedPtr sub_odometry_;
  rclcpp::Subscription<AccelWithCovarianceStamped>::SharedPtr sub_acceleration_;
  rclcpp::Subscription<TrackedObjects>::SharedPtr sub_tracked_objects_;
  rclcpp::Subscription<TrafficLightGroupArray>::SharedPtr sub_traffic_signals_;
  rclcpp::Subscription<LaneletRoute>::SharedPtr sub_route_;
  rclcpp::Subscription<LaneletMapBin>::SharedPtr sub_map_;
  rclcpp::Subscription<TurnIndicatorsReport>::SharedPtr sub_turn_indicators_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Cached messages
  Odometry::SharedPtr current_odometry_;
  AccelWithCovarianceStamped::SharedPtr current_acceleration_;
  TrackedObjects::SharedPtr current_objects_;
  TrafficLightGroupArray::SharedPtr current_traffic_signals_;
  LaneletRoute::SharedPtr current_route_;
  LaneletMapBin::SharedPtr current_map_;
  TurnIndicatorsReport::SharedPtr current_turn_indicators_;

  // Preprocessing state
  std::deque<geometry_msgs::msg::Pose> ego_history_odom_;
  std::deque<TurnIndicatorsReport> turn_indicators_history_;
  std::optional<AgentData> agent_data_{std::nullopt};
  std::optional<AgentData> last_ego_centric_agent_data_{std::nullopt};
  std::map<lanelet::Id, TrafficSignalStamped> traffic_light_id_map_;
  std::unique_ptr<preprocess::LaneSegmentContext> lane_segment_context_;
  VehicleInfo vehicle_info_;

  // Normalization
  utils::NormalizationMap normalization_map_;

  // Postprocessing parameters
  int64_t velocity_smoothing_window_ = 8;
  double stopping_threshold_ = 0.3;
  bool predict_neighbor_trajectory_ = true;

  // Anytime CSV logging (node-level, with ego pose)
  std::ofstream anytime_csv_file_;
  bool anytime_csv_header_written_ = false;

  // State
  bool is_map_loaded_ = false;
};

}  // namespace autoware::diffusion_planner_onnx_split

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner_onnx_split::DiffusionPlannerNode)
