#pragma once
#include <Arduino.h>

// ── Actuator types (servo, relay, buzzer, LED) ────────────────────────
enum ActuatorType : uint8_t {
  ACTUATOR_NONE    = 0,
  ACTUATOR_SERVO   = 1,   // 0-180 degrees
  ACTUATOR_RELAY   = 2,   // 0=off 1=on
  ACTUATOR_BUZZER  = 3,   // 0=silent 1=active
  ACTUATOR_LED     = 4,   // 0=off 0-255=brightness (analog write)
  ACTUATOR_TYPE_COUNT = 5,
};

// ── Servo configuration ───────────────────────────────────────────────
#define MAX_SERVOS 4

struct ServoConfig {
  uint8_t gpio;        // GPIO pin
  uint8_t channel;     // LEDC channel (0-7)
  bool    attached;    // is this slot in use?
  int     current_angle; // last commanded angle (0-180)
};

// 9g micro servo (SG90-class) PWM parameters
// 50Hz, 20ms period. 1ms=0°, 1.5ms=90°, 2ms=180°
// duty = (pulse_ms / 20.0) * 65536  (16-bit LEDC)
static constexpr uint32_t SERVO_FREQ_HZ     = 50;
static constexpr uint8_t  SERVO_RESOLUTION  = 16;  // 16-bit duty
static constexpr uint32_t SERVO_DUTY_MIN    = 3277;   // ~1ms  = 0°  (3277/65536 * 20ms ≈ 1.0ms)
static constexpr uint32_t SERVO_DUTY_MAX    = 6554;   // ~2ms  = 180° (6554/65536 * 20ms ≈ 2.0ms)
static constexpr uint32_t SERVO_DUTY_NEUTRAL = 4915;  // ~1.5ms = 90°

// Default GPIO pins for servos on Freenove ESP32-S3 WROOM
// These avoid flash/PSRAM pins (GPIO 26-32) and strapping pins (GPIO 0)
static constexpr uint8_t DEFAULT_SERVO_PINS[MAX_SERVOS] = { 4, 5, 6, 7 };

// ── Convert angle (0-180) to LEDC duty ────────────────────────────────
static inline uint32_t servo_angle_to_duty(int angle) {
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;
  // Linear interpolation: 0° → SERVO_DUTY_MIN, 180° → SERVO_DUTY_MAX
  return SERVO_DUTY_MIN + (uint32_t)((SERVO_DUTY_MAX - SERVO_DUTY_MIN) * angle) / 180;
}

// ── Policy-to-actuator mapping ────────────────────────────────────────
// The policy decides servo positions based on sensor state.
// The LM NEVER directly controls actuators — it only provides advisory text.
struct ActuatorCommand {
  uint8_t  servo_channel;
  int      servo_angle;
  bool     servo_move;   // should we move this servo?
  const char *reason;    // why the actuator was commanded
};

// Default servo mapping for a vent/door servo on channel 0:
// - Normal/cold/dry/no-claim → 0° (vent closed, safe position)
// - Hot/humid/air quality/gas/CO2 → 180° (vent open, emergency position)
// - Motion/door/vibration → 90° (alert position)
static inline ActuatorCommand policy_to_servo(uint8_t policy_decision) {
  ActuatorCommand cmd = { 0, 0, false, "none" };
  switch (policy_decision) {
    // Emergency: open vent
    case 3:  // POLICY_HOT_HUMID
    case 8:  // POLICY_AIR_QUALITY_HIGH
    case 11: // POLICY_CO2_HIGH
    case 12: // POLICY_GAS_DETECTED
    case 9:  // POLICY_MOTION_DETECTED
    case 15: // POLICY_VIBRATION_HIGH
      cmd.servo_angle = 180;
      cmd.servo_move = true;
      cmd.reason = "emergency_vent_open";
      break;
    // Alert position
    case 10: // POLICY_DOOR_OPEN
    case 17: // POLICY_HIGH_SOUND
      cmd.servo_angle = 90;
      cmd.servo_move = true;
      cmd.reason = "alert_position";
      break;
    // Safe: vent closed
    case 0:  // POLICY_NORMAL
    case 1:  // POLICY_HOT (partial — open vent partially)
      cmd.servo_angle = (policy_decision == 1) ? 90 : 0;
      cmd.servo_move = (policy_decision == 1);
      cmd.reason = policy_decision == 1 ? "partial_vent" : "safe_vent_closed";
      break;
    // Degraded / missing / stale: close vent (safe default)
    case 2:  // POLICY_HUMID
    case 6:  // POLICY_STALE
    case 7:  // POLICY_MISSING
    case 21: // POLICY_DEGRADED
      cmd.servo_angle = 0;
      cmd.servo_move = true;
      cmd.reason = "safe_vent_closed";
      break;
    // Other conditions: no servo action
    default:
      cmd.servo_move = false;
      break;
  }
  return cmd;
}