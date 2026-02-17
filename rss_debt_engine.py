import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches

# 设置中文字体（如果报错，改为 'DejaVu Sans'）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RSSLaboratory:
    """
    曲率债务实验室 v2.0
    验证：洛伦兹变换下的债务重分配 + 区间压缩动态
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.35)

        # 初始参数
        self.kappa = -0.234
        self.v = 0.0  # 速度 (c为单位)
        self.f_rest = 1.0  # 静止系刷新率
        self.n = 10  # 离散度
        self.gamma = 1.0

        # 计算初始债务
        self.debt_data = self.calculate_debt_landscape()

        # 绘制初始图像
        self.im = self.ax.imshow(self.debt_data, cmap='RdYlGn_r', aspect='auto',
                                 extent=[0, 10, 0, 10], origin='lower')
        self.colorbar = plt.colorbar(self.im, ax=self.ax, label='Debt D = |κ|f²n²')

        # 添加理论公式文本
        self.text_info = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      fontsize=11, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.ax.set_xlabel('Space Dimension (n)')
        self.ax.set_ylabel('Time Refresh Rate (f)')
        self.ax.set_title('RSS Debt Landscape: Lorentz Transformation vs Interval Compression')

        # 添加滑块
        self._setup_sliders()

    def calculate_debt_landscape(self):
        """计算债务景观"""
        f_range = np.linspace(0.01, 2.0, 100)
        n_range = np.linspace(1, 50, 100)
        F, N = np.meshgrid(f_range, n_range)

        # 洛伦兹因子
        if abs(self.v) >= 1:
            self.v = 0.999
        self.gamma = 1 / np.sqrt(1 - self.v ** 2)

        # 变换后的f（时间压缩）
        f_prime = F / self.gamma

        # 变换后的n（空间膨胀）
        n_prime = N * self.gamma

        # 债务计算 D = |κ|f²n²
        debt = abs(self.kappa) * (f_prime ** 2) * (n_prime ** 2)

        # 限制最大值防止颜色溢出
        return np.clip(debt, 0, 100)

    def _setup_sliders(self):
        """设置交互滑块"""
        # 速度滑块
        ax_v = plt.axes([0.2, 0.2, 0.6, 0.03])
        self.slider_v = Slider(ax_v, 'Velocity (v/c)', 0.0, 0.99, valinit=self.v)
        self.slider_v.on_changed(self._update)

        # kappa滑块
        ax_k = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_k = Slider(ax_k, 'Curvature κ', -1.0, 0.0, valinit=self.kappa)
        self.slider_k.on_changed(self._update)

        # 离散度滑块
        ax_n = plt.axes([0.2, 0.10, 0.6, 0.03])
        self.slider_n = Slider(ax_n, 'Base Discreteness n', 1, 30, valinit=self.n)
        self.slider_n.on_changed(self._update)

        # 重置按钮
        ax_reset = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        self.button_reset.on_clicked(self._reset)

        # 模式切换按钮
        ax_mode = plt.axes([0.1, 0.02, 0.15, 0.04])
        self.button_mode = Button(ax_mode, 'Mode: Newton', color='lightblue', hovercolor='0.975')
        self.button_mode.on_clicked(self._toggle_mode)

        self.mode = 'newton'  # newton, interval, ghost

    def _update(self, val):
        """更新函数"""
        self.v = self.slider_v.val
        self.kappa = self.slider_k.val
        self.n = self.slider_n.val

        # 重新计算
        self.debt_data = self.calculate_debt_landscape()
        self.im.set_array(self.debt_data)
        self.im.set_clim(vmin=0, vmax=50)

        # 更新文本
        gamma = self.gamma
        f_transformed = self.f_rest / gamma
        n_transformed = self.n * gamma
        current_debt = abs(self.kappa) * (f_transformed ** 2) * (n_transformed ** 2)

        info_text = (
            f"γ (Lorentz factor): {gamma:.3f}\n"
            f"f' = f/γ = {f_transformed:.3f} Hz\n"
            f"n' = n·γ = {n_transformed:.1f}\n"
            f"D = |κ|f'²n'² = {current_debt:.4f}\n"
            f"\nPhysics Interpretation:\n"
        )

        if self.v < 0.1:
            info_text += "Newton Limit (κ≈0): Classical debt accumulation"
        elif self.v > 0.9:
            info_text += "Relativistic Limit: Debt explosion due to γ³ factor"
        else:
            info_text += "RSS Transition: Debt reallocating from time to space"

        self.text_info.set_text(info_text)
        self.fig.canvas.draw_idle()

    def _reset(self, event):
        """重置按钮"""
        self.slider_v.reset()
        self.slider_k.reset()
        self.slider_n.reset()

    def _toggle_mode(self, event):
        """切换存在模式"""
        if self.mode == 'newton':
            self.mode = 'interval'
            self.button_mode.label.set_text('Mode: Interval')
            # 区间模式：一次性展开所有n
            self.slider_n.set_val(50)  # 最大展开
            self.slider_v.set_val(0.0)  # 静止但展开
        elif self.mode == 'interval':
            self.mode = 'ghost'
            self.button_mode.label.set_text('Mode: Ghost')
            # 幽灵模式：f->0, n->0
            self.slider_n.set_val(1)
            self.slider_v.set_val(0.0)
            self.slider_k.set_val(-0.234)
        else:
            self.mode = 'newton'
            self.button_mode.label.set_text('Mode: Newton')
            self.slider_n.set_val(10)
            self.slider_v.set_val(0.0)

    def add_trajectory_overlay(self):
        """添加理论轨迹线（区间压缩路径）"""
        # 绘制从牛顿到相对论的债务轨迹
        v_values = np.linspace(0, 0.99, 100)
        debts = []
        for v in v_values:
            g = 1 / np.sqrt(1 - v ** 2 + 1e-10)
            f_p = 1.0 / g
            n_p = 10 * g
            d = abs(-0.234) * (f_p ** 2) * (n_p ** 2)
            debts.append(d)

        # 归一化到坐标系显示
        # 这个可以在扩展版本中加入
        pass


def demonstrate_third_law():
    """
    第三定律演示：区间压缩的动态可视化
    展示 n 从 1 到 50 时债务的平方增长
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_values = np.linspace(1, 50, 100)
    kappa_values = [-0.1, -0.234, -0.5, -1.0]
    f = 1.0

    # 左图：债务曲线
    for kappa in kappa_values:
        debts = [abs(kappa) * (f ** 2) * (n ** 2) for n in n_values]
        ax1.plot(n_values, debts, label=f'κ={kappa}')

    ax1.set_xlabel('Discrete Degree n (Number of Branches)')
    ax1.set_ylabel('Debt D = |κ|f²n²')
    ax1.set_title('Third Law: Interval Compression Cost')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 对数坐标显示平方爆炸

    # 右图：洛伦兹变换下的债务重分配
    v_values = np.linspace(0, 0.99, 100)
    debts_lorentz = []

    for v in v_values:
        gamma = 1 / np.sqrt(1 - v ** 2 + 1e-10)
        # 假设固有时债务为 D0 = 0.234 * 1 * 100 = 23.4
        D0 = 0.234 * 1 * 100
        # 变换后 D' = D0 * γ^3 （因为 f->f/γ, n->nγ, 所以 D' = κ(f/γ)²(nγ)² = κf²n²γ² = D0γ²）
        D_prime = D0 * (gamma ** 2)
        debts_lorentz.append(D_prime)

    ax2.plot(v_values, debts_lorentz, 'r-', linewidth=2, label='Debt after Lorentz Boost')
    ax2.axhline(y=23.4, color='k', linestyle='--', label='Rest Frame Debt (D0)')
    ax2.set_xlabel('Velocity v/c')
    ax2.set_ylabel('Debt D')
    ax2.set_title('Lorentz Transformation: Debt Reallocation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.suptitle('RSS Validation: Debt Mechanics', fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    print("正在启动曲率债务实验室...")
    print("操作说明：")
    print("1. 拖动 Velocity 滑块观察洛伦兹变换下的债务重分配")
    print("2. 拖动 κ 滑块改变曲率档位")
    print("3. 点击 Mode 按钮切换牛顿/区间/幽灵三种存在模式")
    print("4. 红色区域=高债务危险区，绿色=膜态安全区")

    # 启动交互式实验室
    lab = RSSLaboratory()

    # 同时显示静态分析图
    demonstrate_third_law()

    plt.show()

    print("\n实验结束。结论：")
    print("1. 当 v->c (速度接近光速)，γ->∞，债务 D 随 γ² 爆炸增长")
    print("2. 这就是为何'闪现'（高v）需要预付巨额债务（区间压缩）")
    print("3. 幽灵模式（低f,低n）始终保持绿色（低债务）")
