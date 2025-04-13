import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use("TkAgg")


def plot_csv(exp: str, create_legend_fig=False) -> None:
    if exp.__contains__('bcw'):
        title = 'Breast Cancer Wisconsin'
    elif exp.__contains__('heart_uci'):
        title = 'UCI Heart Disease'
    elif exp.__contains__('actg'):
        title = 'ACTG 320 Clinical Trial'
    elif exp.__contains__('stroke'):
        title = 'Stroke Prediction'
    elif exp.__contains__('thyroid'):
        title = 'Thyroid Cancer Risk'
    else:
        title = exp

    path = f'experiments/{exp}/experiment.csv'
    df = pd.read_csv(path)

    x = df["generation"]
    y_columns = ["baseline_random", "baseline_best", "gen_rand", "gen_seq"]
    BASELINE_RANDOM = "#B99095"
    BASELINE_BEST = "#FCB5AC"
    GEN_SEQ = "#3D5B59"
    GEN_RAND = 'olive'

    colors = {
        "baseline_best": BASELINE_BEST,
        "baseline_random": BASELINE_RANDOM,
        "gen_seq": GEN_SEQ,
        "gen_rand": GEN_RAND
    }

    plt.figure(figsize=(10, 6))
    for col in y_columns:
        plt.plot(x, df[col], marker='o', label=col, color=colors[col])

    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Best Dataset (Avg. Metrics)", fontsize=16)
    plt.title(title, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'experiments/{exp}/{exp}.png')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.close()

    if create_legend_fig:
        label_mapping = {
            "baseline_random": "Baseline - Random Selection",
            "baseline_best": "Baseline - Best of the Original Synthetic Datasets",
            "gen_rand": "Genetic Algorithm - Random Initialization",
            "gen_seq": "Genetic Algorithm - Sequential Initialization"
        }
        new_labels = [label_mapping.get(lbl, lbl) for lbl in labels]

        legend_fig = plt.figure(figsize=(10, 6))

        legend_fig.legend(handles, new_labels, loc="center", ncol=1, fontsize=14)
        legend_fig.suptitle("Legend", fontsize=20)
        legend_fig.canvas.draw()
        plt.axis("off")
        legend_fig.savefig('legend.png')
        plt.close()


def plot_avg_fitness() -> None:
    experiments = ['exp_actg', 'exp_bcw', 'exp_heart_uci', 'exp_stroke', 'exp_thyroid_cancer']
    dfs_rand = {}
    dfs_seq = {}

    for exp in experiments:
        path_rand = f'experiments/{exp}/log_gen_rand.csv'
        path_seq = f'experiments/{exp}/log_gen_seq.csv'
        dfs_rand[exp] = pd.read_csv(path_rand)
        dfs_seq[exp] = pd.read_csv(path_seq)

    exp2name = {
        'exp_bcw': 'Breast Cancer Wisconsin',
        'exp_heart_uci': 'UCI Heart Disease',
        'exp_actg': 'ACTG 320 Clinical Trial',
        'exp_stroke': 'Stroke Prediction',
        'exp_thyroid_cancer': 'Thyroid Cancer Risk'
    }

    name2color = {
        'Breast Cancer Wisconsin': '#264653',
        'UCI Heart Disease': '#2a9d8f',
        'ACTG 320 Clinical Trial': '#e9c46a',
        'Stroke Prediction': '#f4a261',
        'Thyroid Cancer Risk': '#e76f51'
    }

    df_rand = pd.DataFrame()
    df_seq = pd.DataFrame()
    df_rand['generation'] = dfs_rand[experiments[0]]['generation']
    df_seq['generation'] = dfs_seq[experiments[0]]['generation']

    for exp in experiments:
        df_rand[exp2name[exp]] = dfs_rand[exp]['average_fitness']
        df_seq[exp2name[exp]] = dfs_seq[exp]['average_fitness']

    # Determine the overall y-axis limits
    combined_min = min(df_rand.drop(columns='generation').min().min(),
                       df_seq.drop(columns='generation').min().min())
    combined_max = max(df_rand.drop(columns='generation').max().max(),
                       df_seq.drop(columns='generation').max().max())

    f, axs = plt.subplots(1, 2, figsize=(20, 7))

    # Random Initialization Plot
    axs[0].set_title('Random Initialization', fontsize=20)
    x = df_rand["generation"]
    y_columns = [col for col in df_rand.columns if col != 'generation']
    for col in y_columns:
        axs[0].plot(x, df_rand[col], label=col, marker='o', markersize=4, color=name2color[col])
    axs[0].set_xlabel("Generation", fontsize=16)
    axs[0].set_ylabel("Avg. Population Fitness", fontsize=16)
    axs[0].set_ylim(combined_min, combined_max + 0.01)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    # Sequential Initialization Plot
    axs[1].set_title('Sequential Initialization', fontsize=20)
    x = df_seq["generation"]
    y_columns = [col for col in df_seq.columns if col != 'generation']
    for col in y_columns:
        axs[1].plot(x, df_seq[col], label=col, marker='o', markersize=4, color=name2color[col])
    axs[1].set_xlabel("Generation", fontsize=16)
    axs[1].set_ylabel("Avg. Population Fitness", fontsize=16)
    axs[1].set_ylim(combined_min, combined_max + 0.01)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].legend(loc='lower right')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('avg_fitness.png', dpi=600)


def combine_figures() -> None:
    imgA = Image.open("experiments/exp_bcw/exp_bcw.png")
    imgB = Image.open("experiments/exp_heart_uci/exp_heart_uci.png")
    imgC = Image.open("experiments/exp_actg/exp_actg.png")
    imgD = Image.open("experiments/exp_stroke/exp_stroke.png")
    imgE = Image.open("experiments/exp_thyroid_cancer/exp_thyroid_cancer.png")
    img_legend = Image.open('legend.png')

    grid = [
        [imgA, imgB],
        [imgC, imgD],
        [imgE, img_legend]
    ]

    num_rows = 3
    num_cols = 2

    # Compute maximum width for each column
    col_widths = []
    for col in range(num_cols):
        widths = []
        for row in range(num_rows):
            if grid[row][col] is not None:
                widths.append(grid[row][col].width)
        col_widths.append(max(widths) if widths else 0)

    # Compute maximum height for each row
    row_heights = []
    for row in range(num_rows):
        heights = []
        for col in range(num_cols):
            if grid[row][col] is not None:
                heights.append(grid[row][col].height)
        row_heights.append(max(heights) if heights else 0)

    # Determine overall canvas dimensions
    canvas_width = sum(col_widths)
    canvas_height = sum(row_heights)

    # Create a new white canvas for the combined image
    combined = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Paste each image into its cell in the grid
    y_offset = 0
    for r in range(num_rows):
        x_offset = 0
        for c in range(num_cols):
            img = grid[r][c]
            if img is not None:
                combined.paste(img, (x_offset, y_offset))
            x_offset += col_widths[c]
        y_offset += row_heights[r]

    combined.save("combined_result.png", dpi=(600, 600))
    print('Combined image saved as "path/to/combined_result.png"')

