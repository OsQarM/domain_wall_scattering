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

    for i in range(N):
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
    ax.set_xlabel(r'Time $t/\tau_{\mathrm{transfer}}$', fontsize=12)
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
        vmin=-1,
        vmax=1,
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