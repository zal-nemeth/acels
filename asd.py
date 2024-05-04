# Define the function to compute the new hex with opacity
def hex_with_opacity(original_hex, alpha):
    r = int(original_hex[1:3], 16)
    g = int(original_hex[3:5], 16)
    b = int(original_hex[5:7], 16)

    # Assuming white background (255, 255, 255)
    r_new = round(r * alpha + 255 * (1 - alpha))
    g_new = round(g * alpha + 255 * (1 - alpha))
    b_new = round(b * alpha + 255 * (1 - alpha))

    return f"#{r_new:02x}{g_new:02x}{b_new:02x}"


# Input dictionary of hex values
hexes = {
    "Non-Quant Adam": "#1f77b4",
    "Quant Adam": "#ff7f0e",
    "Non-Quant Adamax": "#2ca02c",
    "Quant Adamax": "#d62728",
    "Non-Quant Nadam": "#9467bd",
    "Quant Nadam": "#8c564b",
    "Non-Quant RMSprop": "#e377c2",
    "Quant RMSprop": "#7f7f7f",
}

# New dictionary for updated hex values
new_hexes = {}

# Convert hex values according to their type (Non-Quant and Quant)
for key, value in hexes.items():
    if "Quant" in key:
        new_hexes[key] = hex_with_opacity(value, 0.6)
    else:
        new_hexes[key] = hex_with_opacity(value, 0.8)

# Print the updated dictionary
print(new_hexes)
