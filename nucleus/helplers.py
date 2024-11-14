import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


def display_pdd(dir="./dataset/PDD.csv"):
    # Load data
    df = pd.read_csv(dir, index_col=False)
    X = np.arange(1, 11.05, 0.05)
    Y = df.iloc[:, 0].to_numpy()
    Z = df.iloc[:, 1:].to_numpy()

    # Create the original meshgrid
    X, Y = np.meshgrid(X, Y)

    # Flatten the data for interpolation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    # Create a finer grid for smoother plot
    xi = np.linspace(X.min(), X.max(), 400)  # Increase to 400 points along X-axis
    yi = np.linspace(Y.min(), Y.max(), 400)  # Increase to 400 points along Y-axis
    X_smooth, Y_smooth = np.meshgrid(xi, yi)

    # Interpolate Z to the finer grid
    Z_smooth = griddata((x_flat, y_flat), z_flat, (X_smooth, Y_smooth), method="cubic")

    # Plot the smoother 3D surface
    fig = plt.figure(figsize=(30, 60))
    ax = fig.add_subplot(111, projection="3d")

    # Create the smooth surface plot
    surf = ax.plot_surface(
        X_smooth, Y_smooth, Z_smooth, cmap="viridis", edgecolor="none"
    )

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.1, aspect=7)

    # Set axis labels with increased font size
    ax.set_xlabel("Energy [MeV]", fontsize=20)
    ax.set_ylabel("Depth [cm]", fontsize=20)
    ax.set_zlabel("Normalized Dose", fontsize=20)

    # Set view angle
    ax.view_init(elev=60, azim=45)

    plt.show()
