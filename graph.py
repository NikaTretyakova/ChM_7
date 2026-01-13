import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

def load_parameters():
    """Загрузка параметров"""
    search_paths = [
        Path("C:/Users/Veronika/Desktop/ChM_laba7/parameters.txt"),
        Path("./parameters.txt"),
    ]
   
    for path in search_paths:
        if path.exists():
            print(f"Найден файл параметров: {path}")
            params = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                       
                        if key == 'output_dir':
                            params[key] = value.strip('"\'')
                        else:
                            try:
                                if '.' in value:
                                    params[key] = float(value)
                                else:
                                    params[key] = int(value)
                            except ValueError:
                                params[key] = value
           
            if 'output_dir' not in params:
                params['output_dir'] = "C:\\Users\\Veronika\\Desktop\\ChM_laba7"
           
            params['output_dir'] = str(Path(params['output_dir']).resolve())
            return params
   
    print("Файл parameters.txt не найден! Использую параметры по умолчанию.")
    return {
        'N': 1024,
        'A': 2.15,
        'B': 0.18,
        'w2': 185,
        'output_dir': "C:\\Users\\Veronika\\Desktop\\ChM_laba7"
    }
def load_partial_reconstruction_data(wavelet_name, level, output_dir):
    """Загрузка данных частичного восстановления из новых файлов"""
    if level == 0:
        file_path = Path(output_dir) / "signal.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'value' in df.columns:
                return df['value'].values, df['index'].values
            elif 'z' in df.columns:
                return df['z'].values, df['index'].values
            elif 'signal' in df.columns:
                return df['signal'].values, df['index'].values
    else:
        file_path = Path(output_dir) / f"{wavelet_name}_partial_reconstruction_P_minus_{level}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'value' in df.columns:
                return df['value'].values, df['index'].values
            elif 'z' in df.columns:
                return df['z'].values, df['index'].values
            elif 'signal' in df.columns:
                return df['signal'].values, df['index'].values
            elif len(df.columns) == 2:
                col = [c for c in df.columns if c != 'index'][0]
                return df[col].values, df['index'].values

        stage_file = Path(output_dir) / f"{wavelet_name}_stage{level}.csv"
        if stage_file.exists():
            df = pd.read_csv(stage_file)
            recon_data = df[df['type'] == 'reconstructed']
            if len(recon_data) > 0:
                return recon_data['value'].values, recon_data['index'].values
   
    print(f"    Не удалось загрузить P_{-level}(z)")
    return None, None

def load_coefficients_data(wavelet_name, level, coeff_type, output_dir):
    """Загрузка коэффициентов ψ и φ - ПРОСТОЙ МЕТОД"""
    file_path = None
   
    if coeff_type == 'psi':
        file_path = Path(output_dir) / f"{wavelet_name}_Q_minus_{level}.csv"

        if not file_path.exists():
            file_path = Path(output_dir) / f"{wavelet_name}_psi_level{level}.csv"
   
    elif coeff_type == 'phi':
        file_path = Path(output_dir) / f"{wavelet_name}_P_minus_{level}_approx.csv"

        if not file_path.exists():
            file_path = Path(output_dir) / f"{wavelet_name}_phi_level{level}.csv"
   
    if file_path is None or not file_path.exists():
        print(f"    Файл не найден: {file_path}")
        return None, None
   
    try:
        df = pd.read_csv(file_path)
       
        for col in df.columns:
            if col.lower() in ['index', 'k', 'i', 'j', 'idx']:
                continue
            try:
                sample_values = df[col].head(5).astype(float)
                values = df[col].astype(float).values

                index_cols = ['index', 'k', 'i', 'j', 'idx']
                for idx_col in index_cols:
                    if idx_col in df.columns:
                        indices = df[idx_col].values
                        return values, indices

                return values, np.arange(len(values))
            except:
                continue

        if len(df.columns) >= 2:
            values = df.iloc[:, 1].astype(float).values
            indices = np.arange(len(values))
            return values, indices
       
        return None, None
           
    except Exception as e:
        print(f"    Ошибка чтения файла {file_path}: {e}")
        return None, None

def plot_single_stage(wavelet_name, level, params):
    """Построение 4 графиков для одного этапа"""
    if params is None:
        return
   
    output_dir = Path(params.get('output_dir', '.'))
    N = params.get('N', 1024)

    recon_values, recon_indices = load_partial_reconstruction_data(wavelet_name, level, output_dir)

    psi_values, psi_indices = load_coefficients_data(wavelet_name, level, 'psi', output_dir)
    phi_values, phi_indices = load_coefficients_data(wavelet_name, level, 'phi', output_dir)

    original_file = Path(output_dir) / "signal.csv"
    if original_file.exists():
        df_original = pd.read_csv(original_file)
        if 'value' in df_original.columns:
            original_values = df_original['value'].values
            original_indices = df_original['index'].values
        elif 'z' in df_original.columns:
            original_values = df_original['z'].values
            original_indices = df_original['index'].values
        else:
            # Если есть только один столбец кроме index
            value_col = [c for c in df_original.columns if c != 'index'][0]
            original_values = df_original[value_col].values
            original_indices = df_original['index'].values
    else:
        print(f"    Файл signal.csv не найден!")
        return
   
    if recon_values is None and level > 0:
        print(f"    Нет данных восстановления для этапа {level}")
        return

    fig = plt.figure(figsize=(16, 10), dpi=100)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    if level == 0:
        title = f'ВЕЙВЛЕТ: {wavelet_name.upper()} | ИСХОДНЫЙ СИГНАЛ P_0(z)'
    else:
        title = f'ВЕЙВЛЕТ: {wavelet_name.upper()} | ЭТАП: P_{{-{level}}}(z) = P_{{-{level}}}(z)_approx + Q_{{-{level}}}(z)'
   
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(original_indices, original_values, 'b-', linewidth=1.2, alpha=0.8)
    ax1.set_title('1. ИСХОДНЫЙ СИГНАЛ P_0(z)', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, N)

    ax2 = fig.add_subplot(gs[0, 1])
    if psi_values is not None and len(psi_values) > 0:
        if psi_indices is None:
            psi_indices = np.arange(len(psi_values))

        if not np.all(np.diff(psi_indices) >= 0):
            sort_idx = np.argsort(psi_indices)
            psi_indices = psi_indices[sort_idx]
            psi_values = psi_values[sort_idx]
       
        ax2.plot(psi_indices, psi_values, 'r-', linewidth=1.0, alpha=0.8, marker='o', markersize=3)
       
        ax2.set_xlabel('k (индекс)', fontsize=11)
        ax2.set_ylabel(f'Q_{{-{level}}}(k)', fontsize=11)
       
        ax2.set_title(f'2. ВЫСОКОЧАСТОТНЫЕ КОЭФФИЦИЕНТЫ Q_{{-{level}}}(z)',
                     fontsize=12, fontweight='bold', pad=10)
       
        ax2.grid(True, alpha=0.3, linestyle='--')
        if len(psi_indices) > 0:
            ax2.set_xlim(min(psi_indices) - 0.5, max(psi_indices) + 0.5)
        else:
            ax2.set_xlim(-0.5, len(psi_values) - 0.5)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        if len(psi_values) > 0:
            stats_text = f'Мин: {np.min(psi_values):.4f}\nМакс: {np.max(psi_values):.4f}\nСредн: {np.mean(psi_values):.4f}'
            ax2.text(0.02, 0.98, stats_text,
                    transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'2. ВЫСОКОЧАСТОТНЫЕ КОЭФФИЦИЕНТЫ Q_{{-{level}}}(z)',
                     fontsize=12, fontweight='bold', pad=10)

    ax3 = fig.add_subplot(gs[1, 0])
    if recon_values is not None and len(recon_values) > 0:
        if level == 0:
            color = 'b'
        else:
            color = 'g'
       
        ax3.plot(recon_indices if recon_indices is not None else np.arange(len(recon_values)),
                 recon_values, color=color, linewidth=1.5, alpha=0.8)
       
        ax3.set_xlabel('j (отчеты)', fontsize=11)
        ax3.set_ylabel('z(j)', fontsize=11)
        if level == 0:
            ax3.set_title(f'3. ИСХОДНЫЙ СИГНАЛ P_0(z)',
                         fontsize=12, fontweight='bold', pad=10)
        else:
            ax3.set_title(f'3. ЧАСТИЧНО ВОССТАНОВЛЕННЫЙ СИГНАЛ\nP_{{-{level}}}(z)',
                         fontsize=12, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(0, len(recon_values))
    else:
        if level == 0:
            ax3.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title(f'3. ИСХОДНЫЙ СИГНАЛ P_0(z)',
                         fontsize=12, fontweight='bold', pad=10)
        else:
            ax3.text(0.5, 0.5, 'Нет данных восстановления', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title(f'3. ЧАСТИЧНО ВОССТАНОВЛЕННЫЙ СИГНАЛ\nP_{{-{level}}}(z)',
                         fontsize=12, fontweight='bold', pad=10)

    ax4 = fig.add_subplot(gs[1, 1])
    if phi_values is not None and len(phi_values) > 0:
        if phi_indices is None:
            phi_indices = np.arange(len(phi_values))
       
        if not np.all(np.diff(phi_indices) >= 0):
            sort_idx = np.argsort(phi_indices)
            phi_indices = phi_indices[sort_idx]
            phi_values = phi_values[sort_idx]
       
        ax4.plot(phi_indices, phi_values, 'g-', linewidth=1.0, alpha=0.8, marker='o', markersize=3)
       
        ax4.set_xlabel('k (индекс)', fontsize=11)
        ax4.set_ylabel(f'P_{{-{level}}}(k)_approx', fontsize=11)
       
        ax4.set_title(f'4. НИЗКОЧАСТОТНЫЕ КОЭФФИЦИЕНТЫ P_{{-{level}}}(z)_approx',
                     fontsize=12, fontweight='bold', pad=10)
       
        ax4.grid(True, alpha=0.3, linestyle='--')
        if len(phi_indices) > 0:
            ax4.set_xlim(min(phi_indices) - 0.5, max(phi_indices) + 0.5)
        else:
            ax4.set_xlim(-0.5, len(phi_values) - 0.5)
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
       
        if len(phi_values) > 0:
            stats_text = f'Мин: {np.min(phi_values):.4f}\nМакс: {np.max(phi_values):.4f}\nСредн: {np.mean(phi_values):.4f}'
            ax4.text(0.02, 0.98, stats_text,
                    transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax4.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title(f'4. НИЗКОЧАСТОТНЫЕ КОЭФФИЦИЕНТЫ P_{{-{level}}}(z)_approx',
                     fontsize=12, fontweight='bold', pad=10)
   
    plt.tight_layout(rect=[0, 0, 1, 0.96])
   
    if level == 0:
        output_path = Path(output_dir) / f"{wavelet_name}_P0.png"
    else:
        output_path = Path(output_dir) / f"{wavelet_name}_P_minus_{level}.png"
   
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_comparison_plot(wavelet_name, params):
    """Построение сравнения всех частичных восстановлений на одном графике"""
    output_dir = Path(params.get('output_dir', '.'))

    reconstructions = {}

    original_file = Path(output_dir) / "signal.csv"
    if original_file.exists():
        df = pd.read_csv(original_file)
        if 'value' in df.columns:
            reconstructions[0] = df['value'].values
        elif 'z' in df.columns:
            reconstructions[0] = df['z'].values
        elif len(df.columns) == 2:
            col = [c for c in df.columns if c != 'index'][0]
            reconstructions[0] = df[col].values

    for level in range(1, 5):
        recon_values, _ = load_partial_reconstruction_data(wavelet_name, level, output_dir)
        if recon_values is not None:
            reconstructions[level] = recon_values
   
    if len(reconstructions) < 2:
        print(f"    Недостаточно данных для сравнения {wavelet_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
   
    for i, (level, signal) in enumerate(reconstructions.items()):
        if level == 0:
            label = 'P_0(z) (исходный)'
            linewidth = 1.5
            alpha = 0.8
            linestyle = '-'
        else:
            label = f'P_{{-{level}}}(z)'
            linewidth = 1.2
            alpha = 0.7
            linestyle = '-'
       
        indices = np.arange(len(signal))
        ax.plot(indices, signal, color=colors[i % len(colors)],
                linewidth=linewidth, alpha=alpha, label=label,
                linestyle=linestyle)
   
    ax.set_xlabel('j (отчеты)', fontsize=12)
    ax.set_ylabel('z(j)', fontsize=12)
    ax.set_title(f'СРАВНЕНИЕ ЧАСТИЧНЫХ ВОССТАНОВЛЕНИЙ\nВейвлет: {wavelet_name.upper()}',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, len(list(reconstructions.values())[0]))
   
    plt.tight_layout()

    output_path = Path(output_dir) / f"{wavelet_name}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Сохранен график сравнения: {wavelet_name}_comparison.png")
    plt.close(fig)

def check_file_structure(output_dir, wavelet_name):
    """Проверка структуры файлов для отладки"""
    print(f"\n    Структура файлов для {wavelet_name}:")

    for level in range(1, 5):
        file_path = Path(output_dir) / f"{wavelet_name}_partial_reconstruction_P_minus_{level}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"      P_{-level}: {file_path.name}")
                print(f"        Столбцы: {list(df.columns)}")
                print(f"        Размер: {len(df)} строк")
                if len(df) > 0:
                    print(f"        Первые 3 значения: {df.iloc[:3, 1].values[:3]}")
            except:
                print(f"      P_{-level}: {file_path.name} - ошибка чтения")

    for level in range(1, 5):
        file_path = Path(output_dir) / f"{wavelet_name}_Q_minus_{level}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"      Q_{-level}: {file_path.name}")
                print(f"        Столбцы: {list(df.columns)}")
                print(f"        Размер: {len(df)} строк")
            except:
                print(f"      Q_{-level}: {file_path.name} - ошибка чтения")
       
        file_path = Path(output_dir) / f"{wavelet_name}_P_minus_{level}_approx.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"      P_{-level}_approx: {file_path.name}")
                print(f"        Столбцы: {list(df.columns)}")
                print(f"        Размер: {len(df)} строк")
            except:
                print(f"      P_{-level}_approx: {file_path.name} - ошибка чтения")

def plot_all_stages_for_wavelet(wavelet_name, params):
    """Построение всех этапов для одного вейвлета"""
    print(f"\n  Вейвлет {wavelet_name}:")
    print("  " + "-" * 48)
   
    output_dir = Path(params.get('output_dir', '.'))
   
    check_file_structure(output_dir, wavelet_name)
   
    print(f"    Этап 0 (исходный сигнал)...")
    plot_single_stage(wavelet_name, 0, params)
   
    # Затем частичные восстановления
    for level in range(1, 5):
        print(f"    Этап {level} (P_{-level}(z))...")
        plot_single_stage(wavelet_name, level, params)
   
    print(f"    График сравнения...")
    plot_comparison_plot(wavelet_name, params)




def main():
    params = load_parameters()
   
    output_dir = Path(params['output_dir'])
    N = params.get('N', 1024)
    wavelets = ['Haar', 'Shannon', 'Daubechies_D6']
   
    if not (output_dir / "signal.csv").exists():
        print("  ОШИБКА: Файл signal.csv не найден!")
        print("  Сначала запустите C++ программу для генерации данных.")
        return
   
    for wavelet in wavelets:
        plot_all_stages_for_wavelet(wavelet, params)

if __name__ == "__main__":
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as e:
        print(f"Ошибка: {e}")
        print("\nУстановите необходимые библиотеки:")
        print("pip install numpy pandas matplotlib")
        sys.exit(1)

    plt.style.use('seaborn-v0_8-darkgrid')
   
    main()
