#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
#include <direct.h>
#define mkdir _mkdir
#else
#include <sys/stat.h>
#endif

using namespace std;
using cd = complex<double>;

const double PI = 3.14159265358979323846;
const double EPS = 1e-10;

bool directoryExists(const string& path) {
#ifdef _WIN32
    struct _stat info;
    return _stat(path.c_str(), &info) == 0 && (info.st_mode & _S_IFDIR);
#else
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

bool createDirectory(const string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0;
#else
    return mkdir(path.c_str(), 0755) == 0;
#endif
}

class WaveletAnalyzer {
public:
    enum WaveletType { HAAR, SHANNON, DAUBECHIES_D6 };

private:
    WaveletType type;
    vector<double> h;  
    vector<double> g;  

    void initializeFilters() {
        h.clear();
        g.clear();

        switch (type) {
        case HAAR:
            h = { 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
            g = { 1.0 / sqrt(2.0), -1.0 / sqrt(2.0) };
            break;

        case SHANNON:
            // Простые фильтры для Шеннона (аналогичные Хаару)
            h = { 0.5, 0.5 };
            g = { 0.5, -0.5 };
            break;

        case DAUBECHIES_D6:
        {
            const double rt10 = sqrt(10.0);
            const double a = 1.0 - rt10;
            const double b = 1.0 + rt10;
            const double c = sqrt(5.0 + 2.0 * rt10);
            const double k = sqrt(2.0) / 32.0;

            h = {
                k * (b + c),
                k * (2.0 * a + 3.0 * b + 3.0 * c),
                k * (6.0 * a + 4.0 * b + 2.0 * c),
                k * (6.0 * a + 4.0 * b - 2.0 * c),
                k * (2.0 * a + 3.0 * b - 3.0 * c),
                k * (b - c)
            };

            g.resize(6);
            for (int i = 0; i < 6; i++) {
                g[i] = h[5 - i];
                if ((5 - i) % 2 == 1) {
                    g[i] = -g[i];
                }
            }
            break;
        }
        }
    }

public:
    WaveletAnalyzer(WaveletType wavelet_type) : type(wavelet_type) {
        initializeFilters();
    }

    void dwt(const vector<double>& signal, vector<double>& approx, vector<double>& detail) {
        int M = signal.size();
        int L = h.size();

        approx.resize(M / 2);
        detail.resize(M / 2);

        for (int i = 0; i < M / 2; i++) {
            double a = 0.0, d = 0.0;
            for (int k = 0; k < L; k++) {
                int idx = (2 * i + k) % M;
                if (idx < M) {
                    a += h[k] * signal[idx];
                    d += g[k] * signal[idx];
                }
            }
            approx[i] = a;
            detail[i] = d;
        }
    }

    void idwt(const vector<double>& approx, const vector<double>& detail, vector<double>& signal) {
        int M = approx.size() * 2;
        int L = h.size();

        signal.assign(M, 0.0);

        for (int i = 0; i < M / 2; i++) {
            for (int k = 0; k < L; k++) {
                int idx = (2 * i + k) % M;
                if (idx < M) {
                    signal[idx] += h[k] * approx[i] + g[k] * detail[i];
                }
            }
        }
    }

    void performPartialReconstruction(const vector<double>& signal,
        int max_level,
        vector<vector<double>>& all_details,
        vector<vector<double>>& all_approx,
        vector<vector<double>>& all_partial_reconstructions) {

        all_details.clear();
        all_approx.clear();
        all_partial_reconstructions.clear();

        all_details.resize(max_level + 1);
        all_approx.resize(max_level + 1);
        all_partial_reconstructions.resize(max_level + 1);
        all_partial_reconstructions[0] = signal;
        all_approx[0] = signal;

        vector<double> current_signal = signal;

        for (int j = 1; j <= max_level; j++) {
            if (current_signal.size() < 2) {
                cerr << "Сигнал слишком мал для дальнейшего разложения на уровне " << j << endl;
                break;
            }

            vector<double> approx, detail;
            dwt(current_signal, approx, detail);
            all_details[j] = detail;
            all_approx[j] = approx;
            vector<double> reconstructed;
            idwt(approx, detail, reconstructed);

            all_partial_reconstructions[j] = reconstructed;
            current_signal = approx;
        }
    }
};

vector<double> generateSignal(int N, double A, double B, int w2) {
    vector<double> signal(N, 0.0);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> noise_dist(0.0, 0.05);

    for (int j = 0; j < N; j++) {
        double value = 0.0;

        if (0 <= j && j < N / 4) {
            value = 0.0;
        }
        else if (N / 4 <= j && j <= N / 2) {
            value = A + B * cos(2.0 * PI * w2 * j / N);
        }
        else if (N / 2 < j && j <= 3 * N / 4) {
            value = 0.0;
        }
        else if (3 * N / 4 < j && j < N) {
            value = A + B * cos(2.0 * PI * w2 * j / N);
        }

        value += noise_dist(gen);
        signal[j] = value;
    }

    return signal;
}

void saveSignalToCSV(const string& filename, const vector<double>& signal) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return;
    }

    file << "index,value\n";
    for (size_t i = 0; i < signal.size(); i++) {
        file << i << "," << signal[i] << "\n";
    }
    file.close();
}

void saveCoefficientsToCSV(const string& filename,
    const vector<double>& coeffs) {

    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return;
    }

    file << "index,value\n";
    for (size_t k = 0; k < coeffs.size(); k++) {
        file << k << "," << coeffs[k] << "\n";
    }
    file.close();
}

void savePartialReconstructionsToCSV(const string& filename_prefix,
    const vector<vector<double>>& partial_reconstructions,
    const string& wavelet_name) {

    for (size_t level = 0; level < partial_reconstructions.size(); level++) {
        string filename;
        if (level == 0) {
            filename = filename_prefix + "\\signal.csv";
        }
        else {
            filename = filename_prefix + "\\" + wavelet_name +
                "_partial_reconstruction_P_minus_" + to_string(level) + ".csv";
        }

        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Ошибка открытия файла: " << filename << endl;
            continue;
        }

        file << "index,value\n";
        const auto& signal = partial_reconstructions[level];
        for (size_t i = 0; i < signal.size(); i++) {
            file << i << "," << signal[i] << "\n";
        }
        file.close();

        if (level == 0) {
            cout << "   Сохранен P_0(z) (исходный сигнал): " << signal.size() << " точек\n";
        }
        else {
            cout << "   Сохранен P_{-" << level << "}(z): " << signal.size() << " точек\n";
        }
    }
}

int main() {
    setlocale(0, "");
    int n = 10;
    int N = (int)pow(2.0, n);
    double A = 2.15;
    double B = 0.18;
    int w2 = 185;

    string output_dir = "C:\\Users\\Veronika\\Desktop\\ChM_laba7";

    if (!directoryExists(output_dir)) {
        cout << "Создание директории: " << output_dir << "...\n";
        if (!createDirectory(output_dir)) {
            cerr << "Ошибка создания директории. Использую текущую директорию.\n";
            output_dir = ".";
        }
    }

    vector<double> signal = generateSignal(N, A, B, w2);

    string signal_file = output_dir + "\\signal.csv";
    saveSignalToCSV(signal_file, signal);
    cout << "   Сигнал сохранен в: " << signal_file << "\n\n";

    vector<WaveletAnalyzer::WaveletType> wavelet_types = {
        WaveletAnalyzer::HAAR,
        WaveletAnalyzer::SHANNON,
        WaveletAnalyzer::DAUBECHIES_D6
    };

    vector<string> wavelet_names = { "Haar", "Shannon", "Daubechies_D6" };

    const int MAX_LEVEL = 4;

    for (size_t w_idx = 0; w_idx < wavelet_types.size(); w_idx++) {
        string wavelet_name = wavelet_names[w_idx];
        cout << " Анализ с вейвлетом " << wavelet_name << "...\n";

        try {
            WaveletAnalyzer analyzer(wavelet_types[w_idx]);

            vector<vector<double>> all_details;
            vector<vector<double>> all_approx;
            vector<vector<double>> all_partial_reconstructions;

            analyzer.performPartialReconstruction(signal, MAX_LEVEL,
                all_details, all_approx,
                all_partial_reconstructions);

            cout << "   Частичные восстановления:\n";
            savePartialReconstructionsToCSV(output_dir, all_partial_reconstructions, wavelet_name);

            for (int level = 1; level <= MAX_LEVEL; level++) {
                string detail_file = output_dir + "\\" + wavelet_name +
                    "_Q_minus_" + to_string(level) + ".csv";
                saveCoefficientsToCSV(detail_file, all_details[level]);

                string approx_file = output_dir + "\\" + wavelet_name +
                    "_P_minus_" + to_string(level) + "_approx.csv";
                saveCoefficientsToCSV(approx_file, all_approx[level]);

                cout << "   Уровень " << level << ": "
                    << all_details[level].size() << " деталей, "
                    << all_approx[level].size() << " аппроксимаций\n";
            }

            cout << "   Анализ " << wavelet_name << " завершен.\n\n";

        }
        catch (const exception& e) {
            cerr << "   Ошибка: " << e.what() << "\n\n";
        }
    }

    string params_file = output_dir + "\\parameters.txt";
    ofstream params(params_file);
    if (params.is_open()) {
        params << "N=" << N << "\n";
        params << "A=" << A << "\n";
        params << "B=" << B << "\n";
        params << "w2=" << w2 << "\n";
        params << "n=" << n << "\n";
        params << "output_dir=" << output_dir << "\n";
        params << "max_level=" << MAX_LEVEL << "\n";
        params.close();
    }

    return 0;
}
