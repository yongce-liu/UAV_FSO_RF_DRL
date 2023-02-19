import numpy as np
from scipy import optimize, special


def fso_eqn(x):
    return 1 / x - np.exp(-x) / (1 - np.exp(-x)) - fso_intensity_alpha_ratio


# FSO channel
fso_path_coe = {'clear_air':0.43e-3, 'haze': 4.2e-3, 'light_fog': 20e-3, 'moderate_fog': 42.2e-3, 'heavy_fog':125e-3}
fso_visibility = 0.5  # km
fso_ref_visibility = 550  # nm
fso_transmission_wavelength = 1550  # nm
# C_n_2  m^(-2/3)
# 1.7e-14 near the ground during the daytime
# 8.4e-15 near the ground during the night
# 1e-13 strong turbulence to 1e-17 weak. often 1e-25 average
fso_C_0_2 = 1.7e-14  # m^(-2/3)
# fso_sigma_2 = (2 * np.pi / fso_transmission_wavelength) ** (7 / 6) * fso_C_0_2
# fso_sigma2_plane = 0.307 * fso_sigma_2
# fso_sigma2_spherical = 0.124 * fso_sigma_2
# the initial beam width in m
fso_point_loss_omega_0 = 0.25e-3
# # Variance of the random fluctuation of th UAV beam orientation in rad
# fso_point_loss_sigma_o = 0.3e-3
# # Variance of the random fluctuation of the UAV position in m
# fso_point_loss_sigma_p = 1e-2
# the aperture radius of the receiver in m
fso_receiver_r = 10e-2
# the center of the beam footprint is outside the receiver lens in m
fso_receiver_u = 14e-2
fso_intensity_alpha_ratio = 0.25
fso_unique_mu = optimize.fsolve(fso_eqn, np.array([1]))[0]
# On the Capacity of Free-Space Optical Intensity Channels
# when FSO_ALPHA --> (0, 0.5)
if 0 < fso_intensity_alpha_ratio < 0.5:
    fso_k1 = np.exp(2 * fso_intensity_alpha_ratio * fso_unique_mu) / (2 * np.pi * np.exp(1)) \
             * ((1 - np.exp(-fso_unique_mu) / fso_unique_mu) ** 2) / fso_intensity_alpha_ratio ** 2
# when FSO_ALPHA --> [0.5, 1]
elif 0.5 <= fso_intensity_alpha_ratio <= 1:
    fso_k1 = 1 / (2 * np.pi * np.exp(1)) / fso_intensity_alpha_ratio ** 2
# fso_power = 10  # dBm allowed average-power
fso_noise = -30  # dBm
fso_bandwidth = 100  # MHz
# RF channel
# 光速
light_speed = 3e8
# 载波频率 carrier frequency
rf_f = 2e9  # Hz
rf_eta_los = 0.1  # dB
rf_eta_nlos = 21  # dB
rf_rician_K_db = 15  # dB
rf_bandwidth = 100  # MHz
# rf_power = 20  # dBm
rf_noise = -75  # dBm
cars_height = 2  # m


# On the Capacity of Free-Space Optical Intensity Channels
# Multi-UAV Trajectory Optimization Considering Collisions in FSO Communication Networks
def get_fso_gain(nlos_flag: bool, uav_pos: np.ndarray, distance: np.ndarray, car_pos: np.ndarray) -> np.ndarray:
    """
    :param nlos_flag: 是否被障碍物遮挡
    :param uav_pos: 无人机坐标 [m] (3,)
    :param distance: 无人机与地车辆之间的距离 [m]
    :return: fso 信道增益
    """
    shape = distance.shape
    # path loss
    # # Kruse model
    # assert fso_visibility >= 0
    # if fso_visibility >= 50:
    #     delta = 1.6
    # elif (fso_visibility >= 6) and (fso_visibility < 50):
    #     delta = 1.3
    # else:
    #     delta = 0.585 * fso_visibility ** (1 / 3)
    # # extinction coefficient dB/km
    # L_sca = 3.91 / fso_visibility * (fso_transmission_wavelength / fso_ref_visibility) ** (-delta)
    # h_l =  10 ** (-L_sca * distance * 1e-4)
    h_l = 10 ** (-fso_path_coe['haze'] * distance * 0.1)

    # atmospheric turbulence log-normal model
    # BER Performance of Free-Space Optical Transmission with Spatial Diversity
    # sigma = fso_sigma2_plane * distance ** (11 / 6)
    sigma_2 = 0.003
    h_a = np.random.lognormal(mean=-2*sigma_2, sigma=2*np.sqrt(sigma_2), size=shape)

    # point errors
    # 计算传播后 beam 宽度
    # Statistical Modeling of the FSO Fronthaul Channel for UAV - Based Communications
    C_n_2 = fso_C_0_2 * np.exp(-uav_pos[-1] / 100)  # m^(-2/3)
    k = 2 * np.pi / (fso_transmission_wavelength * 1e-9)
    rho_l = (0.55 * C_n_2 * distance * k ** 2) ** (-3 / 5)
    # [m]
    omega_d = fso_point_loss_omega_0 * \
              np.sqrt(1 + (1 + (2 * fso_point_loss_omega_0 ** 2 / rho_l ** 2)) *
                      ((fso_transmission_wavelength * 1e-9 * distance) /
                       (np.pi * fso_point_loss_omega_0 ** 2)) ** 2)
    
    # 计算相关参数
    r_dis = np.linalg.norm(car_pos[:, 0:-1] - uav_pos[0:-1], axis=1)
    phi_beam_z = np.arctan(r_dis / (uav_pos[-1] - cars_height))

    # if uav_pos[1] >= 0:
    theta_beam_xy_x = np.arccos(np.abs(uav_pos[0] - car_pos[:, 0]) / r_dis)
    # else:
    #     theta_beam_xy_x = -np.arccos(uav_pos[0] - car_pos[:, 0] / r_dis)

    niu_1 = np.sqrt(np.pi / 2) * fso_receiver_r / omega_d
    niu_2 = np.abs(np.sin(phi_beam_z) * np.cos(theta_beam_xy_x)) * niu_1

    A_0 = special.erf(niu_1) * special.erf(niu_2)

    k_g = np.sqrt(np.pi) / 4 * (special.erf(niu_1) / (niu_1 * np.exp(-niu_1 ** 2)) + special.erf(niu_2) / (
            np.sin(phi_beam_z) ** 2 * np.cos(theta_beam_xy_x) ** 2 * niu_2 * np.exp(-niu_2) ** 2))

    h_p = A_0 * np.exp(2 * fso_receiver_u ** 2 / (-k_g * omega_d ** 2))
    # print(h_a)

    # 平方后
    h = (h_l * h_a * h_p) ** 2 * (1 - nlos_flag)
    # print(los_flag)

    return h


# Trajectory Design for UAV-Based Internet of Things Data Collection: A Deep Reinforcement Learning Approach
# Optimal LAP Altitude for Maximum Coverage
def get_rf_gain(nlos_flag: bool, distance: np.ndarray) -> np.ndarray:
    """
    :param nlos_flag: 视距类型, 分为 视距: LOS, 非视距: N-LOS
    :param distance: 无人机与车辆之间的距离, 类型为一维数组 [m]
    :return: 信道增益
    """
    num = distance.shape[0]
    # coef
    PL = 20 * np.log10(distance) + 20 * np.log10(rf_f) + 20 * np.log10(4 * np.pi / light_speed)
    # small-scale -- https://zhuanlan.zhihu.com/p/378334372
    # NLOS--Rayleigh fading
    h_rayleigh = np.sqrt(1 / 2) * (np.random.randn(num) + np.random.randn(num) * 1j)

    # LOS--Rician fading with 15-dB Rician factor
    K = 10 ** (rf_rician_K_db / 10)
    h_rician = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * h_rayleigh
    # when LOS
    # rician fading
    path_loss_los = PL + rf_eta_los
    h_small_scale_los = h_rician
    # rayleigh fading
    path_loss_nlos = PL + rf_eta_nlos
    h_small_scale_nlos = h_rayleigh

    path_loss = path_loss_los * (1 - nlos_flag) + path_loss_nlos * nlos_flag
    h_small_scale = h_small_scale_los * (1 - nlos_flag) + h_small_scale_nlos * nlos_flag
    # 平方后
    h_gain = 10 ** (-path_loss / 10) * np.abs(h_small_scale) ** 2

    return h_gain


# On the Capacity of Free-Space Optical Intensity Channels
def get_capacity(mode: str, tx_power: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """
    :param mode: 计算信道容量的模式, 仅允许 FSO 信道 和 RF 信道, 字符串格式
    :param tx_power: 发送的平均功率, [dBm] 10 * log10(x/1000 [mw])
    :param gain: 相应的信道增益 ndarray
    :return: 返回信道容量 [Mbps]
    """
    if mode == "FSO":
        snr = fso_k1 * (10 ** ((tx_power - fso_noise) * 2 / 10)) * gain
        # in Mbps
        rate = fso_bandwidth / 2 * np.log2(1 + snr)
    elif mode == "RF":
        snr = 10 ** ((tx_power - rf_noise) / 10) * gain
        # in Mbps
        rate = rf_bandwidth * np.log2(1 + snr)
    else:
        raise ValueError("暂时仅有针对 FSO 和 RF 的计算")

    return rate


def power_distribute(p_rf: float, h_rf: np.ndarray,
                     p_fso: np.ndarray, h_fso: np.ndarray,
                     target_rate: float):
    # fso channel
    need_power_fso = 5 * np.log10((2 ** (2 * target_rate / fso_bandwidth) - 1) / (fso_k1 * h_fso + 1e-100)) + fso_noise
    power_fso = np.clip(need_power_fso, -np.inf, p_fso)  # in dBm
    rate_fso = get_capacity(mode="FSO", tx_power=power_fso, gain=h_fso)
    # rf channel
    delta_rate = target_rate - rate_fso + 1e-10
    need_power_rf = 10 * np.log10((2 ** (delta_rate / rf_bandwidth) - 1) / h_rf) + rf_noise
    # a = 10 ** (need_power_rf * 0.1)
    # b = np.sum(a)
    # c = 10 * np.log10(b)
    if 10 * np.log10(np.sum(10 ** (need_power_rf * 0.1))) <= p_rf:
        power_rf = need_power_rf  # in dBm
    else:
        temp_power = p_rf
        power_rf = np.ones_like(h_rf) * -1e12  # in dBm
        # 优先为信道状态好的提供, 升序排列
        index_r = np.argsort(h_rf)[::-1]
        i = 0
        while temp_power > -10:
            index = index_r[i]
            temp = need_power_rf[index]  # in dBm
            temp = np.clip(temp, -np.inf, temp_power)  # in dBm
            power_rf[index] = temp
            # print(temp)
            temp_power = 10 * np.log10(10 ** (0.1 * temp_power) - 10 ** (0.1 * temp) + 1e-12)  # in dBm
            i = i + 1

    rate_rf = get_capacity(mode="RF", tx_power=power_rf, gain=h_rf)

    return rate_rf, power_rf, rate_fso, power_fso


if __name__ == "__main__":
    np.random.seed(1)
    gain_rf = get_rf_gain(np.array([False]), np.array([400]))
    print(gain_rf)
    # gain_rf = 1e-9 * np.ones(shape=(4,))
    print(get_capacity(mode="RF", tx_power=20 + 10 * np.log10(1), gain=gain_rf))
    # gain_fso = get_fso_gain(np.array([False, True]), np.array([20, 30, 100]), np.array([100, 300]))
    # print(gain_fso)
    # gain_fso = 1e-4 * np.ones(shape=(4,))
    # print(get_capacity(mode="FSO", tx_power=5, gain=gain_fso))