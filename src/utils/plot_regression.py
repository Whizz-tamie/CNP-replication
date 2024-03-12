import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

def plot_functions(target_x, target_y, context_x, context_y, pred_y, var, filename='plot.png', task=None):
    plt.figure(figsize=(6, 4))  # Adjust the figure size as needed

    # Plot the true function, predicted mean, and context points
    plt.plot(target_x[0], target_y[0], 'k-', linewidth=2, label='True Function')
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2, label='Predicted Mean')
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10, label='Context Points')

    # Fill the area between the variance bounds with a new color
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.2,
        color='lightblue',  # Use the intended color here
        label='Confidence Interval'
    )

    # Labels and title for better context
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Function Prediction with CNP', fontsize=16)

    # Adjust the y and x axis ticks for readability
    plt.yticks([-2, -1, 0, 1, 2], fontsize=12)
    plt.xticks([-2, -1, 0, 1, 2], fontsize=12)

    # Set the y-axis limits to match your data's scale better
    plt.ylim([-2, 2])

    # Enable the grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add a legend outside of the plot to the right
    plt.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1, 1))

    # Ensure a tight layout to prevent clipping
    plt.tight_layout()

    # Save the plot into the 'results' directory with the specified filename
    # Ensure the 'results' directory exists or adjust the path as needed
    plt.savefig(f'./results/{task}/{filename}')

    # Optional: Display the plot
    # plt.show()
