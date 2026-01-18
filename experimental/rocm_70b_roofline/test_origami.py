import origami

# Problem dimensions
M = 2048
N = 2048
K = 2048
B = 1  # batch size

# Get hardware information for device 0
hardware = origami.get_hardware_for_device(0)

# Print hardware information
print("Hardware Information:")
hardware.print()
print()

# Create a simple tile list with one configuration
# Format: (MT0, MT1, DU, MI0, MI1, MI2, MI3, wgm, scale_block_size_a, scale_block_size_b)
tile_list = [
    (256, 256, 64, 16, 16, 32, 1, 6, 0, 0),
    (128, 128, 32, 16, 16, 32, 1, 6, 0, 0),
    (256, 128, 64, 16, 16, 32, 1, 6, 0, 0),
]

# Select best macro tile size
# Parameters: M, N, K, B, trans_a, trans_b, hardware, tile_list,
#             bits_a, bits_b, bits_d, dtype_a, scale_block_size, threshold, print_debug, wgm
ret = origami.select_best_macro_tile_size(
    M,
    N,
    K,
    B,
    True,   # trans_a (transpose A)
    False,  # trans_b (no transpose B)
    hardware,
    tile_list,
    origami.datatype_to_bits(origami.Half),  # bits_a
    origami.datatype_to_bits(origami.Half),  # bits_b
    origami.datatype_to_bits(origami.Half),  # bits_d
    origami.Half,  # dtype_a
    0,      # scale_block_size
    0.8,    # threshold
    True,   # print debug info
    6,      # wgm
)

print(f"\nBest configuration for [{M}, {N}, {B}, {K}]:")
print(f"  Latency: {ret[0][0]:.3f}")
print(f"  MT0: {ret[0][1]}, MT1: {ret[0][2]}, DU: {ret[0][3]}")
print(f"  MI: {ret[0][4]}x{ret[0][5]}x{ret[0][6]}x{ret[0][7]}")
print(f"\nAll configurations tested:")
for i, config in enumerate(ret):
    print(f"  {i+1}. Latency={config[0]:.3f}, MT=({config[1]},{config[2]},{config[3]}), MI=({config[4]},{config[5]},{config[6]},{config[7]})")
