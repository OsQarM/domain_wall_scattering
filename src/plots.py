import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from IPython.display import Image
import numpy as np

def plot_expectations(data, N, J, V, ti, tf, filepath=None):
    # Create simplified figure with only line plot
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # Custom colors
    first_color = '#1f77b4'  # Blue
    last_color = '#d62728'   # Red
    middle_color = '#4a4a4a' # Dark gray

    # ========== SIMPLIFIED LINE PLOT ==========
    ax = fig.add_subplot(111)

    for i in range(data.shape[1]):
        magn = data[:,i]
        norm_time = np.linspace(ti, tf, len(magn))

        lineprops = {
            'color': first_color if i == 0 else (last_color if i == N-1 else middle_color),
            'lw': 2.5 if i in [0, N-1] else 1.0,
            'alpha': 1.0 if i in [0, N-1] else 0.6,
            'label': r'First spin $(n=0)$' if i == 0 else (r'Last spin $(n={})$'.format(N-1) if i == N-1 else None)
        }

        ax.plot(norm_time, magn, **lineprops)

    # Formatting
    ax.set_xlabel(r'Time [ns]', fontsize=12)
    ax.set_ylabel(r'Magnetization $\langle Z \rangle$', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(ti, tf)
    ax.tick_params(labelsize=10)

    # Clean grid
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')

    # Simplified legend - only show first and last if many spins
    if N > 10:
        # Only show first and last for clarity
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]], 
                  fontsize=10, framealpha=0.9, loc='best')
    else:
        ax.legend(fontsize=9, framealpha=0.9, loc='best')

    # Title
    plt.title(f'<Z> for each spin, J = {J}, V = {V}', fontsize=14, pad=15)

    # Final layout adjustment
    plt.tight_layout()

    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)
        print(f'Figure saved to {filepath}')

    plt.show()
    return



def plot_heatmap(data, N, J, V, ti, tf, filepath=None):
    # Create figure for standalone heatmap
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # ========== HEATMAP PLOT ==========
    ax = fig.add_subplot(111)

    # Heatmap with improved display
    heatmap = ax.imshow(
        data.T,
        aspect='auto',
        cmap='coolwarm',
        vmin=np.min(data),
        vmax=np.max(data),
        extent=(ti, tf, 0, N),
        origin='lower',
        interpolation='none'  # Sharp color boundaries
    )

    # Colorbar with matching style
    cbar = fig.colorbar(heatmap, ax=ax, pad=0.03)
    cbar.set_label(r'$\langle Z \rangle$', fontsize=12, rotation=90, labelpad=15)
    cbar.ax.tick_params(labelsize=10)

    # Clean axis formatting
    ax.set_xlabel(r'Time', fontsize=12)
    ax.set_ylabel('Spin Index ($n$)', fontsize=12)
    ax.set_yticks(np.arange(0.5, N+0.5, 1))
    ax.set_yticklabels(np.arange(N))
    ax.tick_params(labelsize=10)

    # Subtle horizontal guides (reduced visibility)
    for n in range(1, N):
        ax.axhline(n, color='white', lw=0.3, alpha=0.15)

    # Title
    plt.title(f'<Z> Heatmap, J={J}, V={V}', fontsize=14, pad=15)

    # Final layout adjustment
    plt.tight_layout()

    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)
        print(f'Figure saved to {filepath}')

    plt.show()
    return



def plot_correlation_heatmap(data, N, J, V, ti, tf, filepath=None):
    """
    Plot correlation heatmap in the style of particle scattering/tunneling plots
    """
    # Create figure with adjusted proportions
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    # Enhanced heatmap with better colormap for correlations
    heatmap = ax.imshow(
        data.T,
        aspect='auto',
        cmap='RdBu_r',  # Red-Blue diverging colormap
        vmin=-1,
        vmax=1,
        extent=(ti, tf, 0, N),
        origin='lower',
        interpolation='bilinear'  # Smoother interpolation for scattering patterns
    )
    
    # Enhanced colorbar with physics notation
    cbar = fig.colorbar(heatmap, ax=ax, pad=0.03)
    cbar.set_label(r'$\langle \sigma_j^z \sigma_{j+1}^z \rangle_{\mathrm{conn}}$', 
                   fontsize=12, rotation=90, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    # Axis labels in the style of physics publications
    ax.set_xlabel(r'$t \times 10^{-3}$', fontsize=12)
    ax.set_ylabel('Spin Index ($n$)', fontsize=12)
    
    # Set y-ticks to show site indices
    ax.set_yticks(np.arange(0.5, N+0.5, 1))
    ax.set_yticklabels(np.arange(N))
    ax.tick_params(labelsize=10)
    
    # Format x-axis ticks to match reference format (-60, 0, 60 style)
    x_center = (ti + tf) / 2
    x_ticks = [ti, x_center, tf]
    x_tick_labels = [f'{ti}', '0', f'{tf}']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=10)
    
    # Subtle horizontal guides
    for n in range(1, N):
        ax.axhline(n, color='white', lw=0.3, alpha=0.15)
    
    # Add center line at t=0 for reference if it's within the range
    if ti <= 0 <= tf:
        ax.axvline(x=0, color='black', lw=0.5, alpha=0.5, linestyle='--')
    
    # Title
    plt.title(f'Correlation Heatmap, J={J}, V={V}', fontsize=14, pad=15)
    
    # Final layout adjustment
    plt.tight_layout()

    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)
        print(f'Figure saved to {filepath}')

    plt.show()
    return



def plot_V_and_T2_heatmap(x_plot, y_plot, data):

    # ========== CONFIGURATION FOR SCIENTIFIC PLOTS ==========
    # Set publication-quality parameters
    plt.rcParams.update({
        'font.family': 'serif',  # or 'sans-serif' depending on journal
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,  # Base font size
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'legend.fontsize': 10,  # Legend size
        'xtick.labelsize': 10,  # X-tick label size
        'ytick.labelsize': 10,  # Y-tick label size
        'figure.dpi': 300,  # High resolution
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.autolayout': False,  # Use constrained_layout instead
        'mathtext.fontset': 'stix',  # For LaTeX-like math fonts
    })

    # ========== DATA GENERATION (REPLACE WITH YOUR DATA) ==========
    # Create sample data grid
    X, Y = np.meshgrid(x_plot, y_plot)

    # Z data (chose magnitude to be represented)

    Z = data

    # ========== FIGURE INITIALIZATION ==========
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    # Common aspect ratios for papers: 1:1, 4:3, or golden ratio (1.618:1)

    ax.set_xscale('log')

    # ========== CREATE HEATMAP ==========
    # Using pcolormesh (better for irregular grids) or imshow (for regular grids)
    heatmap = ax.pcolormesh(Y, X, Z, 
                        cmap='viridis',  # Consider: 'viridis', 'plasma', 'inferno', 'RdBu_r', 'coolwarm'
                        shading='auto',  # 'auto', 'nearest', 'gouraud'
                        edgecolors='none',  # Remove grid lines between cells
                        linewidth=0,
                        rasterized=True)  # Makes PDF files smaller

    # ========== COLORBAR CONFIGURATION ==========
    cbar = fig.colorbar(heatmap, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Z value (units)', fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    # Optional: set specific colorbar ticks
    # cbar.set_ticks([Z.min(), Z.max()/2, Z.max()])
    # cbar.set_ticklabels(['Low', 'Medium', 'High'])

    # ========== AXES AND LABELS ==========
    ax.set_xlabel('T2/T_t', fontsize=12, labelpad=8)
    ax.set_ylabel('V/λ', fontsize=12, labelpad=8)

    # Optional title (often omitted in papers with figure captions)
    # ax.set_title('Descriptive Title', fontsize=14, pad=12)

    # Set axis limits if needed
    # ax.set_xlim([x.min(), x.max()])
    # ax.set_ylim([y.min(), y.max()])

    # Adjust tick frequency and formatting
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(6))  # ~6 major ticks
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))  # 2 minor ticks per major
    # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # Format tick labels if needed (e.g., scientific notation)
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    # ax.xaxis.get_offset_text().set_fontsize(9)
    # ax.yaxis.get_offset_text().set_fontsize(9)

    # Grid (optional, usually omitted in heatmaps)
    # ax.grid(True, which='major', linestyle='--', alpha=0.3, linewidth=0.5)

    # ========== ADDITIONAL ANNOTATIONS ==========
    # Add text annotations if needed
    # ax.text(0.05, 0.95, 'A', transform=ax.transAxes, 
    #         fontsize=16, fontweight='bold', va='top')

    # Mark specific points or regions
    # ax.scatter([5], [0], color='red', s=50, zorder=5, label='Maximum')
    # ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    # ax.axvline(x=5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # ========== SAVE FIGURE ==========
    # Choose appropriate format based on journal requirements
    save_path = 'heatmap_figure.png'  # or .pdf, .eps, .tiff
    # fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)

    # Also save as vector format for editing
    # fig.savefig('heatmap_figure.pdf', format='pdf', bbox_inches='tight')

    plt.show()

    # ========== ADDITIONAL CONSIDERATIONS ==========
    """
    FOR SCIENTIFIC PUBLICATIONS:

    1. COLOR MAP CHOICE:
    - Sequential data: 'viridis', 'plasma', 'inferno', 'magma'
    - Diverging data: 'RdBu_r', 'coolwarm', 'seismic', 'bwr'
    - Categorical: 'tab20c', 'Set3'
    - Avoid: 'jet', 'rainbow' (perceptually non-uniform)

    2. ACCESSIBILITY:
    - Test if plot is interpretable in grayscale
    - Use colorblind-friendly palettes
    - Consider adding patterns or textures if critical distinctions are needed

    3. DATA ASPECT RATIO:
    - Set equal aspect if x and y have same units: ax.set_aspect('equal')
    - Otherwise: ax.set_aspect('auto') (default)

    4. LEGEND/COLORBAR:
    - Always include units in colorbar label
    - Use appropriate number of significant figures
    - Consider log scale if data spans orders of magnitude:
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=Z.min(), vmax=Z.max())
            heatmap = ax.pcolormesh(..., norm=norm)

    5. ANNOTATIONS:
    - Label subfigures (A, B, C) if part of multi-panel figure
    - Add arrows/text to highlight important features

    6. DATA SOURCE:
    - Consider adding small inset showing data distribution
    - Or marginal histograms along axes

    7. FILE FORMATS:
    - .pdf/.eps for vector graphics (line art, simple heatmaps)
    - .tiff/.png for raster graphics (complex heatmaps, large datasets)
    - Check journal specific requirements

    8. SIZE REQUIREMENTS:
    - Single column: 3.3-3.5 inches wide
    - Double column: 6.5-7 inches wide
    - Height: typically 2/3 to 3/4 of width
    """


def plot_transmission_x_T2(t_values, data, labels):

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for better distinction
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Add dashed horizontal line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='<Z>=1')

    # Plot each line as solid lines
    for i in range(data.shape[1]):
        color = colors[i % len(colors)]
        ax.plot(t_values, data[:,i], 
                label=f'V= {labels[i]:.2f}', 
                color=color, 
                linewidth=2,           # Solid line with thickness 2
                linestyle='-',         # Explicitly set to solid line
                marker='',             # Remove markers for clean solid lines
                alpha=0.8)             # Slight transparency

    # CONFIGURE AXIS BOUNDS - Adjust these values as needed
    # Set x-axis limits

    #ax.set_xlim(min(x), max(x))
    ax.set_xscale("log")


    # Customize the plot
    ax.set_xlabel('T2/T_t', fontsize=12)
    ax.set_ylabel('Min <Z> in last qubit', fontsize=12)
    ax.set_title('Minimum Z after the barrier as function of T2, for some V values', fontsize=14)
    ax.legend(fontsize=10)


    plt.tight_layout()
    plt.show()


def plot_transmission_x_V(v_values, data, labels):

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for better distinction
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Add dashed horizontal line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='<Z>=1')

    # Plot each line as solid lines
    for i in range(data.shape[0]):
        color = colors[i % len(colors)]
        ax.plot(v_values, data[i,:], 
                label=f'T2= {labels[i]:.2f}', 
                color=color, 
                linewidth=2,           # Solid line with thickness 2
                linestyle='-',         # Explicitly set to solid line
                marker='',             # Remove markers for clean solid lines
                alpha=0.8)             # Slight transparency

    # CONFIGURE AXIS BOUNDS - Adjust these values as needed
    # Set x-axis limits

    #ax.set_xlim(min(x), max(x))
    #ax.set_xscale("log")


    # Customize the plot
    ax.set_xlabel('V/lambda', fontsize=12)
    ax.set_ylabel('Min <Z> in last qubit', fontsize=12)
    ax.set_title('Minimum Z after the barrier as function of T2, for some V values', fontsize=14)
    ax.legend(fontsize=10)


    plt.tight_layout()
    plt.show()

