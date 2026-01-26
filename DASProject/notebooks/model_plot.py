import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_das_pipeline_diagram():
    # Setup the figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # --- Color Palette (Matching original image) ---
    c_das = 'black'
    c_green_box = '#5da655'   # Green for processes (Windowing, Distance)
    c_green_dark = '#38761d'  # Darker green for final output
    c_orange_box = '#e69138'  # Orange for Model components
    c_blue_store = '#4a86e8'  # Blue for data/storage
    c_arrow = '#6d9eeb'       # Light blue arrows
    c_train_arrow = '#f6b26b' # Orange dashed arrows for weights
    c_dashed = '#999999'      # Gray for optional/future parts

    # --- Helper Functions ---
    def draw_box(x, y, w, h, color, text, text_color='white', font_size=10, edge_color=None, style='solid'):
        if edge_color is None: edge_color = color
        ls = '-' if style == 'solid' else '--'
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor=edge_color, facecolor=color, linestyle=ls)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                color=text_color, fontsize=font_size, fontweight='bold', wrap=True)
        return rect

    def draw_arrow(x1, y1, x2, y2, color=c_arrow, style='->', lw=2, dashed=False):
        ls = '--' if dashed else '-'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls))

    def draw_database(x, y, w, h, color, text):
        # Cylinder shape approximation
        ellipse_h = h * 0.2
        rect_h = h - ellipse_h
        
        # Bottom ellipse
        b_ell = patches.Ellipse((x + w/2, y), w, ellipse_h, facecolor=color, edgecolor=color)
        # Body
        rect = patches.Rectangle((x, y), w, rect_h, facecolor=color, edgecolor=color)
        # Top ellipse
        t_ell = patches.Ellipse((x + w/2, y + rect_h), w, ellipse_h, facecolor=color, edgecolor='white', lw=1)
        
        ax.add_patch(b_ell)
        ax.add_patch(rect)
        ax.add_patch(t_ell)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # --- 1. DAS INPUT (Far Left) ---
    draw_box(0.5, 3.5, 2.0, 1.2, c_das, "DAS\n(Optic Fiber)", font_size=14)
    
    # --- Phase Divider Line ---
    ax.plot([0, 16], [3.2, 3.2], color='gray', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(14.5, 3.4, "TEST PHASE", color='gray', fontsize=12, fontweight='bold', ha='right')
    ax.text(14.5, 2.9, "TRAIN PHASE", color='gray', fontsize=12, fontweight='bold', ha='right')

    # ==================== TRAIN PHASE (BOTTOM) ====================
    
    # Windowing
    draw_box(3.5, 1.2, 2.0, 1.0, c_green_box, "Space-time\nWindow\nGenerating")
    draw_arrow(2.6, 4.0, 3.5, 1.7, color=c_train_arrow, lw=2) # Arrow from DAS
    ax.text(2.8, 2.5, "Unlabeled\nTraining Data", color='black', fontsize=9, ha='center', backgroundcolor='white')

    # Autoencoder Container (Dashed Orange)
    rect = patches.Rectangle((6.2, 0.5), 3.0, 2.4, linewidth=1.5, edgecolor=c_orange_box, facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(7.7, 2.7, "AutoEncoder (CAE)", color=c_orange_box, fontsize=10, fontweight='bold', ha='center')

    # Encoder -> Latent -> Decoder
    draw_box(6.4, 1.6, 1.2, 0.6, c_green_box, "Encoder", font_size=9) # Encoder
    draw_box(6.4, 0.8, 1.2, 0.6, c_green_box, "Decoder", font_size=9) # Decoder
    
    # Internal flow (Loop)
    draw_arrow(5.5, 1.7, 6.4, 1.9, color=c_train_arrow) # In to Encoder
    draw_arrow(7.0, 1.6, 7.0, 1.4, color=c_train_arrow) # Enc to Dec
    
    # Latent Space / Future Clustering Branch
    draw_arrow(7.6, 1.9, 8.8, 1.9, color=c_dashed, style='->', dashed=True)
    draw_database(8.8, 1.5, 1.2, 0.8, 'gray', "Future:\nClusters")
    ax.text(9.4, 2.5, "(Optional)", color='gray', fontsize=8, ha='center')

    # Loss Calculation
    ax.text(7.8, 1.1, r"$||x_i - y_i||^2$", fontsize=12, color='black') # Loss Math
    draw_arrow(7.0, 0.8, 7.0, 0.6, color=c_train_arrow, style='-[', lw=1) # Loop back
    
    # Validation / Threshold Calculation
    draw_arrow(7.6, 0.9, 8.5, 0.9, color=c_train_arrow)
    draw_box(8.5, 0.5, 2.0, 0.8, c_orange_box, "Validation Set\nThresholding")
    ax.text(9.5, 0.3, "(99.9th Percentile)", color='black', fontsize=8, ha='center')

    # ==================== TEST PHASE (TOP) ====================

    # Windowing
    draw_box(3.5, 4.0, 2.0, 1.0, c_green_box, "Space-time\nWindow\nGenerating")
    draw_arrow(2.5, 4.1, 3.5, 4.5, color=c_arrow) # Arrow from DAS

    # Feature Extraction (Inference)
    rect_top = patches.Rectangle((6.2, 3.8), 2.8, 1.5, linewidth=1.5, edgecolor=c_orange_box, facecolor='none', linestyle='--')
    ax.add_patch(rect_top)
    ax.text(7.6, 5.4, "Reconstruction Model", color=c_orange_box, fontsize=10, fontweight='bold', ha='center')
    
    draw_box(6.4, 4.2, 1.0, 0.8, c_green_box, "Encoder", font_size=9)
    draw_box(7.8, 4.2, 1.0, 0.8, c_green_box, "Decoder", font_size=9)
    
    draw_arrow(5.5, 4.5, 6.4, 4.6, color=c_arrow) # Into Encoder
    draw_arrow(7.4, 4.6, 7.8, 4.6, color=c_arrow) # Enc to Dec

    # Distance / Reconstruction Error
    draw_box(10.0, 4.2, 1.8, 0.8, c_green_box, "Reconstruction\nError")
    
    # Arrow from Decoder to Distance
    draw_arrow(8.8, 4.6, 10.0, 4.6, color=c_arrow)
    
    # Skip connection (Input to Distance) for comparison
    draw_arrow(5.5, 4.8, 10.9, 5.5, color=c_arrow, dashed=True) 
    draw_arrow(10.9, 5.5, 10.9, 5.0, color=c_arrow)
    ax.text(9.5, 5.6, "Original Input", color=c_arrow, fontsize=9)

    # Threshold Check
    draw_arrow(11.8, 4.6, 12.5, 4.6, color=c_arrow)
    ax.text(13.2, 5.1, "Apply Threshold\n($\\tau \\approx 3.70$)", color='black', fontsize=9, ha='center')
    
    # Result Smoothing
    draw_box(12.5, 4.2, 2.0, 0.8, c_green_dark, "Morphological\nClosing")
    
    # Final Output
    draw_arrow(14.5, 4.6, 15.2, 4.6, color=c_arrow)
    
    # ==================== CONNECTIONS BETWEEN PHASES ====================
    
    # Weights Transfer
    draw_arrow(7.0, 2.9, 7.0, 3.8, color=c_train_arrow, dashed=True)
    ax.text(7.1, 3.3, "Trained\nWeights", color='black', fontsize=9)

    # Threshold Transfer
    draw_arrow(9.5, 1.3, 10.9, 4.2, color=c_train_arrow, dashed=True)
    ax.text(10.8, 2.5, "Learned Threshold ($\\tau$)", color='black', fontsize=9, backgroundcolor='white')

    plt.tight_layout()
    plt.savefig('pipeline_diagram_updated.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_das_pipeline_diagram()