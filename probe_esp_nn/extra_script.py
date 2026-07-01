"""
PlatformIO pre-build script for ESP-NN FC probe.
Removes P4 sources (incompatible with Xtensa) and cleans up non-build files.
"""
import os
import shutil

Import("env")

board_config = env.BoardConfig()
chip = board_config.get("build.mcu", "")

if "esp32s3" in chip.lower():
    esp_nn_dir = os.path.join("lib", "esp-nn")
    src_dir = os.path.join(esp_nn_dir, "src")

    # Delete P4 source files (they use RISC-V registers incompatible with Xtensa)
    removed = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if "esp32p4" in f:
                os.remove(os.path.join(root, f))
                removed += 1
    if removed:
        print(f"ESP-NN probe: removed {removed} P4 source files")

    # Remove non-build files
    for junk in ["CMakeLists.txt", "idf_component.yml", "Kconfig.projbuild"]:
        p = os.path.join(esp_nn_dir, junk)
        if os.path.exists(p):
            os.remove(p)
    for junk_dir in ["test_app", "tests"]:
        p = os.path.join(esp_nn_dir, junk_dir)
        if os.path.isdir(p):
            shutil.rmtree(p)

    print(f"ESP-NN probe: configured for S3 SIMD")