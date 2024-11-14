import numpy as np
from scipy.special import erf, erfc

"""
# Given parameter values
F0 = 3.476050438473762e-05
A = 0.007632269114959895
Ep = 9.973782233887997
W = 0.00510221548239472
"""


# Hàm để tính phổ năng lượng
def modify_approximated_landau(E, I0, A, Ep, w):
    z = 2 * (Ep - E) / w
    I = I0 + A * np.exp(-0.5 * np.exp(-z) - 0.5 * z)
    return I


"""
F0 = 4.612883894459182e-05
Ep = 9.973424017645547
W =  0.007547612217305409
A =  0.00460311859861495
"""


# electron_energy_spectrum
def electron_energy_spectrum(E, F0, Ep, W, A):
    z = 2 * (E - Ep) / W
    return F0 + A * np.exp(z + 1 - np.exp(z))


"""
# Given parameter values
y0 = 4.718498022517191e-05
A = 0.0023556867863969906
m = 9.977044130804595
v = 0.2769499175444389
a = 0.0029629696919162025
"""


# Define the pearson_type_iv function with the provided parameters
def pearson_type_iv(x, y0, A, m, v, a):
    # Pearson Type IV distribution
    z = (x - m) / v
    return y0 + A * (1 + z**2 / a) ** (-m) * np.exp(-v * np.arctan(z) / a)


"""
# Given parameter values
mu = 9.975910283681163
sigma = 0.002376525048478938
lambd = 196.44814887845516
y0 = 3.338157277718448e-05
A = 4.899827025476344e-05
"""


# Define the emg_distribution function with the provided parameters
def emg_distribution(E, mu, sigma, lambd, y0, A):
    term1 = (lambd / 2) * np.exp(-lambd * (mu - E - (sigma**2 * lambd) / 2))
    term2 = 1 + erf((mu - E - (sigma**2 * lambd)) / (np.sqrt(2) * sigma))
    return y0 + A * term1 * term2


"""
# Given parameter values
y0 = 3.3381691432545284e-05
A = 4.8998253733318164e-05
xc = 9.975910282735793
w = 0.00237652429080501
t0 = 0.005090397780168652
"""


# Define the gauss_mod_left function with the provided parameters
def gauss_mod_left(x, y0, A, xc, w, t0):
    z = ((x - xc) / w + w / t0) / np.sqrt(2)
    return y0 + 0.5 * A / t0 * (np.exp(0.5 * (w / t0) ** 2 + (x - xc) / t0) * erfc(z))


def generate_training_data(
    num_samples, matrix_R, E_values, parameters, noise_stddev=0.03
):
    """
    Tạo dữ liệu huấn luyện cho mô hình mạng nơ-ron.

    Args:
        num_samples: Số lượng mẫu dữ liệu cần tạo.
        matrix_R: Ma trận được sử dụng để tính toán PDD từ phổ năng lượng.
        E_values: Các giá trị năng lượng được sử dụng để tạo phổ.
        parameters: Một từ điển xác định phạm vi cho mỗi tham số:
            {
                'I0': (giá_trị_min, giá_trị_max),
                'A': (giá_trị_min, giá_trị_max),
                'Ep': (giá_trị_min, giá_trị_max),
                'w': (giá_trị_min, giá_trị_max)
            }
        noise_stddev: Độ lệch chuẩn của nhiễu Gaussian được thêm vào PDD.

    Returns:
        Bộ đôi mảng NumPy: (X_train, Y_train)
            X_train: Dữ liệu đầu vào (các giá trị PDD có nhiễu).
            Y_train: Dữ liệu đầu ra (các giá trị tham số tương ứng).

    Ví dụ:
        parameters = {
        'I0': (0.000001, 0.0003),
        'A': (0.01, 0.1),
        'Ep': (1, 11),
        'w': (0.001, 0.1)
        }

        X_train, Y_train = generate_training_data(
            num_samples=1000,
            matrix_R=matrix_R,  # Your matrix_R
            E_values=E_values,  # Your E_values
            parameters=parameters,
            noise_stddev=0.05
        )
    """

    X_train = []
    Y_train = []

    for _ in range(num_samples):

        # Tạo phổ dựa trên các tham số
        # TODO: Make function adaptable.
        spectrum = modify_approximated_landau(E_values, **parameters)
        spectrum /= np.sum(spectrum)  # Chuẩn hóa phổ

        # Tính toán PDD
        PDD_calculated = np.dot(matrix_R, spectrum)

        # Thêm nhiễu Gaussian
        noise = np.random.normal(0, noise_stddev, PDD_calculated.shape)
        PDD_noisy = PDD_calculated + noise

        # Thêm dữ liệu vào tập huấn luyện
        X_train.append(PDD_noisy)
        Y_train.append(list(parameters.values()))

    return np.array(X_train), np.array(Y_train)
