#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Please replace the following path by your LinLibertine_RB.ttf path
# font_manager.fontManager.addfont('~/Library/Fonts/LinLibertine_RB.ttf')     # register the font

# for font in font_manager.fontManager.ttflist:
#     # view fonts
#     print(font.name, '-', font.fname)


# Setting the font style globally
plt.rcParams.update({
    'font.family': 'serif',
    # 'font.serif': ['Linux Libertine']})
    'font.serif': ['Times New Roman']})

# Data
methods = [
    "iGAD", "iGAD", "HimNet", "HimNet", "TUAF", "TUAF", "GLocalKD", "GLocalKD", "MssGAD", "MssGAD", "OCGIN", "OCGIN",
    "OCGTL", "OCGTL", "GSN", "GSN", "GraphSage", "GraphSage", "GAT", "GAT", "GCN", "GCN", "GRSN", "GRSN"
]
method_names = list(set(methods[::2]))
datasets = ["HSE", "DHFR", "IMDB", "p53", "Reddit", "AIDS", "Proteins_full", "MMP"]
# As we do not get the results for GSN, we use 0.00 as placeholder for it
# GSN starts from 15th position in the following lists
# [(log(#Parameters1), AUC1), (log(#Parameters2), AUC2), ...]
data = {
    "HSE": [24.7, 0.779, 632, 0.596, 581, 0.682, 790, 0.521, 230.8, 0.504, 17.3, 0.642, 104, 0.668, 1.00, 0.626, 5.8, 0.556, 5.5,
            0.534, 2.7, 0.516, 24, 0.758],
    "DHFR": [18.2, 0.575, 602, 0.311, 579, 0.590, 792, 0.614, 230.8, 0.692, 17.3, 0.613, 104, 0.599, 1.00, 0.727, 0.3, 0.535, 2.5,
             0.619, 1.2, 0.583, 16.2, 0.748],
    "IMDB": [None, None, 666, 0.522, None, None, 790, 0.497, 230.8, 0.599, 20, 0.611, 120, 0.605, 1.00, 0.564, 5.8, 0.574, 5.5,
             0.565, 2.7, 0.561, 8.1, 0.749],
    "p53": [24.2, 0.755, 662, 0.652, 577, 0.688, 790, 0.476, 230.8, 0.681, 17.2, 0.624, 103, 0.639, 1.00, 0.663, 5.8, 0.603, 5.5,
            0.561, 2.7, 0.565, 14.4, 0.774],
    "Reddit": [None, None, 4200, 0.756, None, None, 790, 0.784, 230.8, 0.830, 15.7, None, 94, None, 1.00, 0.845, 5.8, 0.530, 5.5,
               0.493, 2.7, 0.742, 3.6, 0.866],
    "AIDS": [18.3, 0.951, 617, 0.725, 564, 0.868, 793, 0.978, 230.7, 0.998, 17.8, 0.919, 101, 0.945, 1.00, 0.994, 0.3, 0.988, 2.6,
             0.989, 1.2, 0.988, 31.5, 0.997],
    "Proteins_full": [21.5, 0.771, 1200, 0.773, 528, 0.767, 818, 0.761, 230.8, 0.781, 15.7, 0.584, 94, 0.608, 1.00, 0.782, 2.8,
                      0.717, 4.2, 0.769, 2, 0.788, 9.1, 0.797],
    "MMP": [24.2, 0.850, 652, 0.699, 577, 0.609, 790, 0.502, 230.8, 0.693, 17.2, 0.764, 103, 0.797, 1.00, 0.788, 5.8, 0.593, 5.5,
            0.548, 2.7, 0.586, 36.3, 0.858]
}

#%%
# Setting larger font sizes
plt.rcParams.update({
    'font.size': 24,  # Default font size for all text
    'axes.labelsize': 24,  # Font size for x and y labels
    'axes.titlesize': 24,  # Font size for the title
    'xtick.labelsize': 20,  # Font size for x-tick labels
    'ytick.labelsize': 20,  # Font size for y-tick labels
    'legend.fontsize': 14  # Font size for legend
})

# Define a distinct color palette
colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    '#6B7CA5',
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",   # blue-teal
    "#E16A79"   # pink
]

markers = [
    '>',
    'o',
    'v',
    'D',
    's',
    'p',
    '+',
    'x',
    '*',
    'v',
    'h',
    'H',
]

markers = {
    method_names[i]: markers[i] for i in range(len(method_names))
}


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('log(#Parameters)')
ax.set_ylabel('AUC')

# Removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)  # 将线条粗细设置为2
ax.spines['bottom'].set_linewidth(3)  # 将线条粗细设置为2

all_x = []
all_y = []

for method_idx, color in zip(range(0, len(methods), 2), colors):
    method_name = methods[method_idx]
    x = []
    y = []
    for dataset in datasets:
        data_x = data[dataset][method_idx]
        data_y = data[dataset][method_idx + 1]
        if data_x is not None and data_y is not None and data_x > 0:  # Ensure positive values for log scale
            x.append(data_x)
            y.append(data_y * 100)
            all_x.append(data_x)
            all_y.append(data_y * 100)

    if x and y:
        ax.scatter(np.log(x), y, label=method_name, color=color, s=300, marker=markers[method_name])
        # ax.plot(np.log(x), y, color=color)  # Connect the dots with unique colors

ax.legend()
# Reducing the margin around the plot
fig.tight_layout(pad=1.0)
plt.show()
fig.savefig("method_performance_plot.png", dpi=150)



# %%
# 选择一个合适的多项式的次数
degree = 5  # 可以调整次数来观察拟合效果

# 使用 numpy.polyfit 来进行多项式拟合，返回的是多项式的系数
X = np.log(np.array(all_x))
Y = np.array(all_y)
coeffs = np.polyfit(X, Y, degree)

# 构建多项式函数
p = np.poly1d(coeffs)

# 绘制拟合的曲线
x_fit = np.linspace(min(X), max(X), 500)
y_fit = p(x_fit)
ax.plot(x_fit, y_fit, color='red', label="Fitted Curve", linewidth=3)

# 根据拟合曲线将数据分为两部分
above_curve = all_y > p(X)  # 找出位于曲线上的散点
below_curve = ~above_curve

ax.legend()
plt.show()
fig.savefig("method_performance_plot_with_decision_boundary.png", dpi=150)


#%%
# 创建新的图形
fig, ax = plt.subplots(figsize=(12, 8))

h_degree = 5
v_degree = 5
# 横向拟合 (x=AUC, y=log(params))
coeffs_h = np.polyfit(Y, X, h_degree)
p_h = np.poly1d(coeffs_h)

# 竖向拟合 (x=log(params), y=AUC) 
coeffs_v = np.polyfit(X, Y, v_degree)
p_v = np.poly1d(coeffs_v)

# 绘制散点图
for method_idx, color in zip(range(0, len(methods), 2), colors):
    method_name = methods[method_idx]
    x = []
    y = []
    for dataset in datasets:
        data_x = data[dataset][method_idx]
        data_y = data[dataset][method_idx + 1]
        if data_x is not None and data_y is not None and data_x > 0:
            y.append(data_y * 100)  # AUC
            x.append(np.log(data_x))  # log(params)
    
    if x and y:
        ax.scatter(x, y, label=method_name, color=color, s=300, marker=markers[method_name])

# 绘制横向拟合曲线
x_fit = np.linspace(min(Y), max(Y), 500)
y_fit_h = p_h(x_fit)
ax.plot(y_fit_h, x_fit, color='red', linewidth=3)

# 绘制竖向拟合曲线
y_fit = np.linspace(min(X), max(X), 500)
x_fit_v = p_v(y_fit)
ax.plot(y_fit, x_fit_v, color='blue', linewidth=3)
# 设置轴标签
ax.set_xlabel('AUC')
ax.set_ylabel('log(#Parameters)')

# 移除上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

# 将图例放在右下角
ax.legend(bbox_to_anchor=(1, 0), loc='lower right')
plt.tight_layout(pad=1.0)
plt.show()
fig.savefig("method_performance_plot_two_fits.png", dpi=150)



# %%