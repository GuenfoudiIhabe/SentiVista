import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Define positions and text for each step
positions = [0.9, 0.7, 0.5, 0.3, 0.1]
steps = [
    'Original:\n"@hater123 http://x.com This is the worst experience ever!"',
    'Remove URLs and @mentions:\n"This is the worst experience ever!"',
    'Remove special characters:\n"This is the worst experience ever"',
    'Remove stopwords (this, is, the):\n"worst experience ever"',
    'Apply stemming/lemmatization:\n"worst experi ever"'
]

# Plot each step as a text box
for pos, text in zip(positions, steps):
    ax.text(0.5, pos, text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', edgecolor='black'))

# Draw arrows between the boxes
for i in range(len(positions) - 1):
    ax.annotate("", xy=(0.5, positions[i] - 0.08), xytext=(0.5, positions[i+1] + 0.08),
                arrowprops=dict(arrowstyle="->", lw=2))

# Set title and display
plt.title("Text Cleaning Pipeline", fontsize=14)
plt.show()
