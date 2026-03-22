#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal core implementation of the Robust Shape Baseline (RSB) wind power
curve model.

This repository intentionally includes only the model logic needed to train and
evaluate the curve-construction pipeline. Dataset files, experiment runners,
project-specific outputs, and manuscript assets are excluded.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import laplace, norm
from sklearn.linear_model import HuberRegressor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PI2MFramework:
    """
    鐗╃悊寮曞鐨勪俊鎭渶澶у寲閲囨牱妗嗘灦

    鏍稿績鎬濇兂锛?
    - 鍒╃敤椋庡姛鐜囩墿鐞嗗叕寮?P = 陆蟻蟺R虏Cp(位)v鲁 鎸囧閲囨牱鍒嗗竷
    - 浣跨敤椴佹M-estimator杩涜鑱氬悎锛堜俊鎭淇濊瘉锛?
    - 灏嗙墿鐞嗙害鏉熻瀺鍏ヨ缁冭繃绋嬶紙闈炰簨鍚庝慨琛ワ級

    鍙傛暟璇存槑锛?
    - air_density: 绌烘皵瀵嗗害 (kg/m鲁), 鏍囧噯鍊?.225
    - rotor_radius: 鍙惰疆鍗婂緞 (m)
    - huber_delta: Huber鎹熷け闃堝€硷紝鎺у埗椴佹鎬?
    - physics_weight: 鐗╃悊鍏堥獙鏉冮噸 [0,1]
    - target_compression: 鐩爣鍘嬬缉姣?
    """

    def __init__(self,
                 air_density=1.225,
                 rotor_radius=None,
                 huber_delta=1.35,
                 physics_weight=0.7,
                 target_compression=0.01,
                 min_core_points=200):
        """
        鍒濆鍖朠I虏M妗嗘灦

        Args:
            air_density: 绌烘皵瀵嗗害 (kg/m鲁)
            rotor_radius: 鍙惰疆鍗婂緞 (m)锛孨one鏃惰嚜鍔ㄦ娴?
            huber_delta: Huber鎹熷け闃堝€?
                - delta=1.35: 95% Gaussian鏁堢巼
                - delta鈫?: 绛変环浜庝腑浣嶆暟锛圠1鑼冩暟锛?
                - delta鈫掆垶: 绛変环浜庡潎鍊硷紙L2鑼冩暟锛?
            physics_weight: 鐗╃悊鍏堥獙鏉冮噸锛孾0,1]
            target_compression: 鐩爣鍘嬬缉姣旓紙閲囨牱鐐?鍘熷鐐癸級
        """
        self.air_density = air_density
        self.rotor_radius = rotor_radius
        self.huber_delta = huber_delta
        self.physics_weight = physics_weight
        self.target_compression = target_compression
        self.min_core_points = min_core_points

        # 鐗╃悊甯告暟棰勮绠?
        self.physical_constant = 0.5 * air_density * np.pi

        # 璁粌鍚庣殑妯″瀷
        self.spline_model = None
        self.training_info = {}

        # 妫€娴嬪埌鐨勫弬鏁?
        self.detected_params = {}

    def _detect_rated_power(self, wind_speeds, powers):
        """
        鑷姩妫€娴嬮瀹氬姛鐜囧弬鏁帮紙宸ョ▼绠€鍖栫増锛?

        鍩轰簬锛?
        1. 楂橀閫熷尯鍩熺殑绋冲畾鍔熺巼鍊?
        2. 鑷€傚簲瀵绘壘鍔熺巼骞冲彴鍖?
        """
        # 妫€娴嬪叧閿閫熺偣锛堝垏鍏ャ€侀瀹氾級
        cut_in, rated = self._detect_critical_wind_speeds(wind_speeds, powers)

        # 鍦ㄩ珮椋庨€熷尯鍩熸壘棰濆畾鍔熺巼锛堝钩鍙板尯锛?
        if rated is not None:
            # 浣跨敤棰濆畾椋庨€熶互涓婄殑鎵€鏈夋暟鎹?
            high_wind_region = wind_speeds >= rated
        else:
            # 鍥為€€锛氫娇鐢?5m/s浠ヤ笂
            high_wind_region = wind_speeds >= 15.0

        if np.sum(high_wind_region) > 50:
            # Use robust high-wind positive power center to avoid underestimating
            # rated power when curtailment/abnormal low-tail points are present.
            high_wind_powers = np.asarray(powers[high_wind_region], dtype=float)
            high_positive = high_wind_powers[high_wind_powers > 0]

            if high_positive.size > 30:
                # Keep upper stable bulk; less sensitive to low-tail drag.
                stable_low = np.percentile(high_positive, 40)
                stable_plateau = high_positive[high_positive >= stable_low]
                if stable_plateau.size > 10:
                    rated_power = float(np.median(stable_plateau))
                else:
                    rated_power = float(np.median(high_positive))
            elif high_positive.size > 0:
                rated_power = float(np.median(high_positive))
            else:
                # Fallback when high-wind region is almost entirely non-positive.
                rated_power = float(np.percentile(high_wind_powers, 95))
        else:
            # 鍥為€€锛氬叏灞€鏈€澶у€奸檮杩?
            rated_power = float(np.percentile(powers, 99))

        # 濡傛灉娌℃湁缁欏嚭鍙惰疆鍗婂緞锛屼娇鐢ㄧ粡楠屽€?
        if self.rotor_radius is None:
            # 鏍规嵁棰濆畾鍔熺巼浼扮畻锛堝父瑙侀鐢垫満缁勶級
            # 1.5MW -> R 鈮?40-45m
            # 1.25MW -> R 鈮?35-40m
            if rated_power > 1400:
                self.rotor_radius = 42.0
            elif rated_power > 1200:
                self.rotor_radius = 38.0
            else:
                self.rotor_radius = 35.0
            logging.info(f"鑷姩妫€娴嬪彾杞崐寰? {self.rotor_radius:.1f} m (based on rated power {rated_power:.0f} kW)")

        cut_in_used = cut_in if cut_in is not None else 3.0
        rated_used = rated if rated is not None else 12.0
        cutin_transition_width = self._estimate_cutin_transition_width(
            wind_speeds, powers, cut_in_used, rated_power
        )

        self.detected_params['rated_power'] = rated_power
        self.detected_params['rotor_radius'] = self.rotor_radius
        self.detected_params['cut_in_wind'] = cut_in_used
        self.detected_params['rated_wind'] = rated_used
        self.detected_params['cutin_transition_width'] = cutin_transition_width

        return rated_power

    def _estimate_plateau_shoulder_width(self, wind_axis, power_axis, rated_wind, plateau_power):
        """
        Estimate a smooth shoulder width (m/s) before the flat rated-power plateau.
        Wider shoulders are used when the pre-rated rise is still steep, which avoids
        a visually abrupt knee near rated wind.
        """
        try:
            w = np.asarray(wind_axis, dtype=float)
            p = np.asarray(power_axis, dtype=float)
            valid = np.isfinite(w) & np.isfinite(p)
            w = w[valid]
            p = p[valid]
            if w.size < 6:
                return 0.8

            # Look at the pre-rated tail where the curve approaches the plateau.
            tail_mask = (w >= rated_wind - 2.0) & (w < rated_wind - 0.1)
            if np.sum(tail_mask) < 5:
                return 0.8

            wt = w[tail_mask]
            pt = p[tail_mask]
            order = np.argsort(wt)
            wt = wt[order]
            pt = pt[order]

            if wt.size >= 7:
                pt = gaussian_filter1d(pt, sigma=1.0, mode='nearest')

            n_tail = min(12, wt.size)
            wt_fit = wt[-n_tail:]
            pt_fit = pt[-n_tail:]

            if np.ptp(wt_fit) < 1e-6:
                return 0.8

            slope = float(np.polyfit(wt_fit, pt_fit, 1)[0])  # kW per m/s
            slope = max(slope, 1.0)

            gap = max(float(plateau_power - pt_fit[-1]), 0.0)  # kW to close
            # Base width from first-order extrapolation, then enlarge slightly for smoothness.
            width = (gap / slope) * 1.35 + 0.45
            return float(np.clip(width, 0.8, 1.8))
        except Exception:
            return 0.8

    def _estimate_cutin_transition_width(self, wind_speeds, powers, cut_in_wind, rated_power):
        """
        Estimate cut-in transition width (m/s) from raw data so that low-speed
        transition is data-driven instead of a fixed hard-coded range.
        """
        try:
            w = np.asarray(wind_speeds, dtype=float)
            p = np.asarray(powers, dtype=float)
            valid = np.isfinite(w) & np.isfinite(p)
            w = w[valid]
            p = p[valid]

            if w.size < 100 or rated_power <= 0:
                return 0.9

            lower = max(0.0, float(cut_in_wind) - 0.2)
            upper = min(float(np.nanmax(w)), float(cut_in_wind) + 3.5)
            if upper - lower < 0.8:
                return 0.9

            local_mask = (w >= lower) & (w <= upper)
            if np.sum(local_mask) < 80:
                return 0.9

            w_local = w[local_mask]
            p_local = p[local_mask]

            edges = np.arange(lower, upper + 0.2, 0.2)
            centers = []
            robust_power = []
            for i in range(len(edges) - 1):
                bm = (w_local >= edges[i]) & (w_local < edges[i + 1])
                if np.sum(bm) < 20:
                    continue

                p_bin = p_local[bm]
                p_pos = p_bin[p_bin > 0]
                if p_pos.size >= 8:
                    # Upper-middle robust center avoids low-tail suppression.
                    p_ref = float(np.percentile(p_pos, 55))
                else:
                    p_ref = float(np.percentile(p_bin, 65))

                centers.append((edges[i] + edges[i + 1]) / 2.0)
                robust_power.append(p_ref)

            if len(centers) < 5:
                return 0.9

            centers = np.asarray(centers, dtype=float)
            robust_power = np.asarray(robust_power, dtype=float)

            if robust_power.size >= 7:
                robust_power = gaussian_filter1d(robust_power, sigma=1.0, mode='nearest')

            # Transition end when low-speed power reaches a small fraction of rated power.
            target_power = max(0.06 * float(rated_power), 60.0)
            valid_search = centers >= (float(cut_in_wind) + 0.25)
            if np.sum(valid_search) < 2:
                return 0.9

            c = centers[valid_search]
            rp = robust_power[valid_search]
            idx = np.where(rp >= target_power)[0]
            if idx.size == 0:
                return 0.9

            transition_end = float(c[idx[0]])
            width = transition_end - float(cut_in_wind)
            return float(np.clip(width, 0.8, 1.8))
        except Exception:
            return 0.9

    def _detect_critical_wind_speeds(self, wind_speeds, powers):
        """
        鑷€傚簲妫€娴嬪垏鍏ラ閫熴€侀瀹氶閫?

        绠楁硶鎬濊矾锛?
        1. 鍒囧叆椋庨€燂細鍔熺巼寮€濮嬫寔缁ぇ浜?鐨勬渶浣庨閫?
        2. 棰濆畾椋庨€燂細鍔熺巼澧為暱鐜囧紑濮嬫樉钁椾笅闄嶏紙杩涘叆骞冲彴鍖猴級

        Returns:
            cut_in_wind, rated_wind
        """
        # 灏嗘暟鎹寜椋庨€熷垎绠憋紝璁＄畻姣忎釜绠辩殑缁熻鐗规€?
        v_min, v_max = wind_speeds.min(), wind_speeds.max()
        bin_edges = np.arange(v_min, v_max + 0.5, 0.5)

        bin_centers = []
        bin_mean_power = []
        bin_std_power = []
        bin_nonzero_ratio = []

        for i in range(len(bin_edges) - 1):
            mask = (wind_speeds >= bin_edges[i]) & (wind_speeds < bin_edges[i + 1])
            if np.sum(mask) > 5:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                p_bin = powers[mask]

                bin_mean_power.append(np.mean(p_bin))
                bin_std_power.append(np.std(p_bin))

                # 闈為浂鍔熺巼鐨勬瘮渚嬶紙鑰冭檻闃堝€硷紝閬垮厤鍣０骞叉壈锛?
                nonzero_threshold = np.percentile(powers[powers > 0], 5) if np.sum(powers > 0) > 10 else 10
                nonzero_ratio = np.sum(p_bin > nonzero_threshold) / len(p_bin)
                bin_nonzero_ratio.append(nonzero_ratio)

        bin_centers = np.array(bin_centers)
        bin_mean_power = np.array(bin_mean_power)
        bin_std_power = np.array(bin_std_power)
        bin_nonzero_ratio = np.array(bin_nonzero_ratio)

        # === 妫€娴嬪垏鍏ラ閫?===
        # 鎵惧埌绗竴涓姛鐜囨寔缁潪闆剁殑鍖洪棿
        cut_in_wind = None
        for i in range(1, len(bin_centers)):
            # 褰撳墠鍜屽悗缁嚑涓閮芥湁鏄捐憲鍔熺巼杈撳嚭
            if bin_centers[i] < 8:  # 鍒囧叆椋庨€熼€氬父鍦? m/s浠ヤ笅
                window_size = min(3, len(bin_centers) - i)
                if np.all(bin_nonzero_ratio[i:i+window_size] > 0.3):
                    # 鎵惧埌绗竴涓繛缁潪闆跺尯闂寸殑璧风偣
                    # 鍚戝墠鍥炴函锛屾壘鍒板姛鐜囩湡姝ｅ紑濮嬬殑鐐?
                    for j in range(max(0, i-2), i+1):
                        if bin_nonzero_ratio[j] > 0.2 and bin_mean_power[j] > 10:
                            cut_in_wind = bin_centers[j]
                            break
                    if cut_in_wind is not None:
                        break

        # 濡傛灉娌℃娴嬪埌锛屼娇鐢ㄧ粺璁℃柟娉?
        if cut_in_wind is None:
            # 鎵惧埌鍔熺巼棣栨瓒呰繃鍏ㄥ眬鏈€澶у姛鐜?%鐨勯閫?
            power_threshold = np.percentile(powers, 95) * 0.05
            for i in range(len(bin_centers)):
                if bin_mean_power[i] > power_threshold and bin_centers[i] < 6:
                    cut_in_wind = bin_centers[i]
                    break

        # 榛樿鍊?
        if cut_in_wind is None:
            cut_in_wind = 3.0
        else:
            # 鍚戜笅鍙栨暣鍒?.5 m/s
            cut_in_wind = round(cut_in_wind * 2) / 2

        # === 妫€娴嬮瀹氶閫燂紙骞冲彴璧风偣锛?==

        # 璁＄畻鍔熺巼鐨勫彉鍖栫巼锛堝鏁帮級
        if len(bin_mean_power) > 5:
            # 浣跨敤宸垎璁＄畻澧為暱鐜?
            power_gradient = np.gradient(bin_mean_power, bin_centers)

            # 骞虫粦姊害
            if len(power_gradient) > 10:
                from scipy.signal import savgol_filter
                window = min(11, len(power_gradient) // 3)
                if window % 2 == 0:
                    window += 1
                if window >= 3:
                    power_gradient = savgol_filter(power_gradient, window, 2)

            # 鎵惧埌姊害鏄捐憲涓嬮檷鐨勭偣锛堥瀹氶閫燂級
            # 鍦ㄥ垏鍏ラ閫熶互涓婃悳绱?
            rated_region = bin_centers > cut_in_wind + 2

            if np.sum(rated_region) > 5:
                # 鎵惧埌姊害闄嶅埌鏈€澶ф搴?0%浠ヤ笅鐨勭涓€鐐?
                max_gradient = np.max(power_gradient[rated_region])
                gradient_threshold = max_gradient * 0.2

                for i in range(len(bin_centers)):
                    if (rated_region[i] and
                        power_gradient[i] < gradient_threshold and
                        bin_mean_power[i] > np.percentile(powers, 80) * 0.5):  # 鍔熺巼宸茬粡杈惧埌杈冮珮姘村钩
                        rated_wind = bin_centers[i]
                        break
                else:
                    # 澶囬€夛細鎵炬爣鍑嗗樊鏈€灏忕殑鐐癸紙骞冲彴鍖烘柟宸皬锛?
                    for i in range(len(bin_centers)):
                        if (rated_region[i] and
                            bin_std_power[i] < np.percentile(bin_std_power[rated_region], 30) and
                            bin_mean_power[i] > np.percentile(powers, 80) * 0.5):
                            rated_wind = bin_centers[i]
                            break
                    else:
                        rated_wind = None
            else:
                rated_wind = None
        else:
            rated_wind = None

        # 榛樿鍊硷細鎵惧姛鐜囪揪鍒?0%鏈€澶у€肩殑椋庨€?
        if rated_wind is None:
            power_90 = np.percentile(powers, 90) * 0.9
            for i in range(len(bin_centers)):
                if bin_mean_power[i] >= power_90:
                    rated_wind = bin_centers[i]
                    break

        # 濡傛灉杩樻槸娌℃壘鍒帮紝浣跨敤鏍囧噯鍊?
        if rated_wind is None:
            rated_wind = 12.0
        else:
            # 鍙栨暣鍒?.5 m/s
            rated_wind = round(rated_wind * 2) / 2

        logging.info(f"鑷€傚簲妫€娴嬪叧閿閫?")
        logging.info(f"  鍒囧叆椋庨€? {cut_in_wind:.1f} m/s")
        logging.info(f"  棰濆畾椋庨€? {rated_wind:.1f} m/s")

        return cut_in_wind, rated_wind

    def _estimate_cp_curve(self, wind_speeds, powers, R=None):
        """
        浼拌鍔熺巼绯绘暟鏇茬嚎 Cp(位)

        鐗╃悊鍏紡锛?
            P = 陆蟻蟺R虏Cp路v鲁
            Cp = 2P / (蟻蟺R虏v鲁)

        绾︽潫锛?
            Cp 鈭?[0, 0.59] (Betz鏋侀檺)
        """
        # 娣诲姞灏忛噺閬垮厤闄ら浂
        v_cubed = wind_speeds**3 + 1e-8

        # 浼拌Cp (濡傛灉R鏈煡锛屼娇鐢ㄩ粯璁ゅ€?0m锛岀粨鏋滃彧鐢ㄤ簬鐩稿姣旇緝)
        R = R or self.rotor_radius or 40.0
        Cp = 2 * powers / (0.5 * self.air_density * np.pi * R**2 * v_cubed)

        # 鐗╃悊绾︽潫锛欳p鍦ㄥ悎鐞嗚寖鍥村唴
        Cp = np.clip(Cp, 0, 0.59)

        # 鎸夐閫熸帓搴忓悗骞虫粦
        sort_idx = np.argsort(wind_speeds)
        v_sorted = wind_speeds[sort_idx]
        Cp_sorted = Cp[sort_idx]

        # 浣跨敤Savitzky-Golay婊ゆ尝鍣ㄥ钩婊?
        if len(Cp_sorted) > 10:
            window = min(15, len(Cp_sorted) // 4)
            if window % 2 == 0:
                window += 1
            if window >= 3:
                Cp_sorted = savgol_filter(Cp_sorted, window, 2)

        return Cp_sorted, v_sorted

    def _compute_information_density(self, wind_speeds, powers):
        """
        璁＄畻淇℃伅瀵嗗害锛堝畾鐞?鍜屽畾鐞?鐨勫簲鐢級

        淇℃伅瀵嗗害 = 伪 脳 鐗╃悊鏇茬巼 + (1-伪) 脳 鏁版嵁鏂瑰樊

        鍏朵腑锛?
        - 鐗╃悊鏇茬巼锛殀Cp''(v)|锛岄珮鏇茬巼鍖哄煙闇€瑕佸瘑闆嗛噰鏍?
        - 鏁版嵁鏂瑰樊锛氬眬閮ㄤ笉纭畾鎬э紝楂樻柟宸渶瑕佹洿澶氶噰鏍?
        """
        # 1. 鐗╃悊鏇茬巼鍒嗛噺
        Cp_sorted, v_sorted_raw = self._estimate_cp_curve(wind_speeds, powers)

        # 瀵归閫熷幓閲嶈仛鍚堬紝閬垮厤閲嶅椋庨€熷鑷磄radient闄ら浂鍜屾洸鐜囧紓甯?
        v_sorted, inverse_idx = np.unique(v_sorted_raw, return_inverse=True)
        if len(v_sorted) >= 3:
            cp_sum = np.bincount(inverse_idx, weights=Cp_sorted)
            cp_count = np.bincount(inverse_idx)
            Cp_unique = cp_sum / np.maximum(cp_count, 1)
        else:
            # 鍥為€€锛氭瀬绔儏鍐典笅淇濇寔鍘熷浼拌
            v_sorted = v_sorted_raw
            Cp_unique = Cp_sorted

        # 璁＄畻Cp瀵归閫熺殑鏇茬巼锛堜娇鐢ㄦ洿椴佹鐨勬柟娉曪級
        try:
            dCp_dv = np.gradient(Cp_unique, v_sorted)
            d2Cp_dv2 = np.gradient(dCp_dv, v_sorted)

            # 澶勭悊NaN鍜孖nf
            d2Cp_dv2 = np.nan_to_num(d2Cp_dv2, nan=0.0, posinf=0.0, neginf=0.0)

            # 鐗╃悊澶嶆潅搴?= |浜岄樁瀵兼暟|
            physical_complexity = np.abs(d2Cp_dv2)

            # 濡傛灉鍏ㄩ儴鏄?鎴栧緢灏忥紝浣跨敤涓€闃跺鏁颁綔涓烘浛浠?
            if physical_complexity.max() < 1e-6:
                physical_complexity = np.abs(dCp_dv)
                physical_complexity = np.nan_to_num(physical_complexity, nan=0.0, posinf=0.0, neginf=0.0)

            # 褰掍竴鍖?
            if physical_complexity.max() > 0:
                physical_complexity /= physical_complexity.max()
            else:
                physical_complexity = np.ones_like(v_sorted)

        except Exception as e:
            logging.warning(f"鐗╃悊澶嶆潅搴﹁绠楀け璐? {e}锛屼娇鐢ㄥ潎鍖€鍒嗗竷")
            physical_complexity = np.ones_like(v_sorted)

        # 2. 鏁版嵁鏂瑰樊鍒嗛噺锛堜俊鎭笉纭畾鎬э級
        local_variance = np.zeros_like(wind_speeds)
        window_width = 1.0  # 1 m/s鑼冨洿

        for i, v in enumerate(wind_speeds):
            neighbors = np.abs(wind_speeds - v) < window_width
            if np.sum(neighbors) > 5:
                local_variance[i] = np.std(powers[neighbors])
            else:
                local_variance[i] = np.std(powers)  # 鍏ㄥ眬鏂瑰樊浣滀负鍥為€€

        # 褰掍竴鍖?
        if local_variance.max() > 0:
            local_variance /= local_variance.max()
        else:
            local_variance = np.ones_like(wind_speeds)

        # 3. Physics-InformedLoss mixing has been removed from the production path.
        # Keep physical_complexity diagnostics for analysis, but sampling density
        # is now purely data-driven by local variance.
        information_density = local_variance

        return information_density, v_sorted, Cp_unique, physical_complexity, local_variance

    def _adaptive_binning(self, wind_speeds, information_density):
        """
        鑷€傚簲鍒嗙锛堝畾鐞?鐨勫簲鐢級

        绛栫暐锛?
        - 楂樹俊鎭瘑搴?鈫?灏忕锛堝瘑闆嗛噰鏍凤級
        - 浣庝俊鎭瘑搴?鈫?澶х锛堢█鐤忛噰鏍凤級
        - 鍒囧叆椋庨€熼檮杩?鈫?寮哄埗瀵嗛泦閲囨牱浠ヤ繚璇佸钩婊戣繃娓?

        绠卞ぇ灏忔槧灏勶細
        - density=1 鈫?min_bin_size
        - density=0 鈫?max_bin_size
        """
        # 鑾峰彇鍒囧叆椋庨€燂紙濡傛灉宸叉娴嬶級
        cut_in_wind = self.detected_params.get('cut_in_wind', 3.0)

        # Define adaptive dense-sampling transition zone around cut-in.
        cutin_transition_width = float(self.detected_params.get('cutin_transition_width', 0.9))
        cutin_transition_width = float(np.clip(cutin_transition_width, 0.8, 1.8))
        transition_start = max(0.0, cut_in_wind - max(0.3, 0.45 * cutin_transition_width))
        transition_end = cut_in_wind + max(1.0, 1.55 * cutin_transition_width)

        v_sorted = np.sort(wind_speeds)
        v_min = v_sorted.min()
        v_max = v_sorted.max()

        # 鏍规嵁鐩爣閲囨牱鏁拌嚜閫傚簲璁剧疆绠卞昂搴︼紝閬垮厤鍥哄畾鍙傛暟瀵艰嚧澶ц妯″潎鍖€鍥為€€銆?
        min_bins_target = int(np.ceil(max(self.min_core_points, len(wind_speeds) * self.target_compression)))
        span = max(v_max - v_min, 1e-6)
        base_step = span / max(min_bins_target, 1)

        # 淇濈暀鈥滃彉瀵嗗害鈥濊兘鍔涳紝鍚屾椂璁╂€讳綋绠辨暟钀藉湪鐩爣閲忕骇銆?
        min_bin_size = max(0.02, 0.6 * base_step)
        max_bin_size = max(min_bin_size * 1.5, 3.0 * base_step)
        transition_bin_size = max(0.02, 0.8 * base_step)

        # 灏嗕俊鎭瘑搴︽槧灏勫埌绠卞ぇ灏?
        density_interp = np.interp(
            v_sorted,
            np.sort(wind_speeds),
            information_density[np.argsort(wind_speeds)]
        )

        bin_sizes = min_bin_size + (max_bin_size - min_bin_size) * (1 - density_interp)

        # === 鍏抽敭鏀硅繘锛氬湪杩囨浮鍖洪棿寮哄埗瀵嗛泦閲囨牱 ===
        # 鎵惧埌杩囨浮鍖洪棿鐨勭储寮?
        transition_mask = (v_sorted >= transition_start) & (v_sorted <= transition_end)
        if np.sum(transition_mask) > 0:
            # 鍦ㄨ繃娓″尯闂翠娇鐢ㄦ洿灏忕殑绠卞昂瀵?
            bin_sizes[transition_mask] = transition_bin_size
            logging.info(
                f"  杩囨浮鍖洪棿瀵嗛泦閲囨牱(鑷€傚簲): [{transition_start:.2f}, {transition_end:.2f}] m/s, "
                f"width={cutin_transition_width:.2f} m/s, 绠卞ぇ灏?{transition_bin_size:.4f} m/s"
            )

        def _build_bin_edges(scale=1.0):
            edges = [v_min]
            current_v = v_min
            for _ in range(10000):
                if current_v >= v_max:
                    break
                nearest_idx = np.searchsorted(v_sorted, current_v, side='left')
                nearest_idx = int(np.clip(nearest_idx, 0, len(v_sorted) - 1))
                bin_size = max(0.02, float(bin_sizes[nearest_idx]) * scale)
                next_v = min(v_max, current_v + bin_size)
                if next_v - edges[-1] <= 1e-9:
                    break
                edges.append(next_v)
                current_v = next_v
            return np.unique(np.asarray(edges))

        # 棣栨鏋勫缓
        bin_edges = _build_bin_edges(scale=1.0)

        # Minimum representation safeguard to keep curve shape stable.
        min_bins = int(np.ceil(max(self.min_core_points, len(wind_speeds) * self.target_compression)))
        current_bins = len(bin_edges) - 1
        if min_bins > 0 and current_bins < min_bins:
            # 缂╁皬鎵€鏈夌瀹斤紝淇濇寔鑷€傚簲褰㈢姸鑰岄潪鐩存帴閫€鍖栨垚鍧囧寑鍒嗙銆?
            scale = max(current_bins / max(min_bins, 1), 0.05)
            refined_edges = _build_bin_edges(scale=scale)
            refined_bins = len(refined_edges) - 1
            min_acceptable_bins = int(max(self.min_core_points, np.floor(0.70 * min_bins)))
            if refined_bins >= min_acceptable_bins:
                bin_edges = refined_edges
                logging.info(
                    f"  Bin count refined adaptively: {current_bins} -> {refined_bins} "
                    f"(target={min_bins}, accepted>= {min_acceptable_bins}, scale={scale:.3f})"
                )
            else:
                # 浠呭湪鏋佺鎯呭喌涓嬫墠浣跨敤鍧囧寑鍥為€€銆?
                uniform_bin_size = max(0.02, span / max(min_bins, 1))
                bin_edges = np.arange(v_min, v_max + uniform_bin_size, uniform_bin_size)
                logging.info(
                    f"  Bin count still low ({refined_bins}); "
                    f"fallback to uniform bins: {len(bin_edges)-1} bins, "
                    f"size≈{uniform_bin_size:.3f} m/s"
                )

        return bin_edges

    def _robust_aggregation_m_estimator(self, wind_speeds, powers, bin_edges):
        """
        椴佹鑱氬悎锛歁-estimator锛堝畾鐞?鐨勫簲鐢級

        鐞嗚鍩虹锛?
        - 鍦↙aplacian鍣０涓嬶紝涓綅鏁版渶澶у寲浼肩劧锛堚墶鏈€灏忓寲鏉′欢鐔碉級
        - Huber浼拌鍣ㄥ湪Gaussian鍣０涓嬫晥鐜?5%锛屼笖瀵圭缇ゅ€奸瞾妫?

        鏂规硶锛?
        - 澶ф牱鏈紙鈮?0锛夛細Huber鍥炲綊
        - 灏忔牱鏈紙<10锛夛細涓綅鏁帮紙鏁板€肩ǔ瀹氾級
        """
        binned_wind = []
        binned_power = []

        for i in range(len(bin_edges) - 1):
            # 绠卞唴鏁版嵁
            mask = (wind_speeds >= bin_edges[i]) & (wind_speeds < bin_edges[i + 1])
            v_bin = wind_speeds[mask]
            p_bin = powers[mask]

            if len(v_bin) < 2:
                continue

            # 绠变腑蹇冮閫?
            v_center = (bin_edges[i] + bin_edges[i + 1]) / 2

            # 椴佹鑱氬悎
            if len(p_bin) >= 10:
                # 浣跨敤Huber浼拌鍣?
                try:
                    # 鏍囧噯鍖杁elta
                    p_std = np.std(p_bin)
                    if p_std > 0:
                        epsilon = self.huber_delta / p_std
                    else:
                        epsilon = 1.5

                    huber = HuberRegressor(
                        epsilon=max(1.1, min(epsilon, 2.0)),
                        alpha=0.0,
                        max_iter=100,
                        warm_start=True
                    )
                    huber.fit(np.zeros_like(p_bin), p_bin)
                    p_center = huber.predict([[0]])[0]
                except:
                    p_center = np.median(p_bin)
            else:
                # 灏忔牱鏈細涓綅鏁版洿绋冲畾
                p_center = np.median(p_bin)

            binned_wind.append(v_center)
            binned_power.append(p_center)

        # 杞崲涓烘暟缁?
        binned_wind = np.array(binned_wind)
        binned_power = np.array(binned_power)

        # === 鍏抽敭锛氬湪閲囨牱闃舵灏卞簲鐢ㄧ墿鐞嗙害鏉燂紙璁烘枃鏍稿績璐＄尞锛?==
        # 纭繚閲囨牱鐐规湰韬弧瓒崇墿鐞嗚寰嬶紝杩欐牱鏍锋潯鎷熷悎鑷劧涔熶細婊¤冻

        # 鑾峰彇妫€娴嬪埌鐨勫叧閿閫燂紙鑷€傚簲锛?
        cut_in_wind = self.detected_params.get('cut_in_wind', 3.0)
        rated_wind = self.detected_params.get('rated_wind', 12.0)

        # === 鍏抽敭鏀硅繘锛氬湪鍒囧叆椋庨€熼檮杩戞坊鍔犲钩婊戣繃娓?===
        # 閬垮厤鍔熺巼鏇茬嚎鍦ㄥ垏鍏ラ閫熷鍑虹幇绐佸厐杞姌
        # 娉ㄦ剰锛氬垏鍏ラ閫熶互涓嬪姛鐜囦弗鏍间负0锛屽钩婊戣繃娓″彧鍦ㄥ垏鍏ラ閫熶箣鍚?

        # Define transition region from cut-in speed with adaptive width.
        transition_start = cut_in_wind
        cutin_transition_width = float(self.detected_params.get('cutin_transition_width', 0.9))
        cutin_transition_width = float(np.clip(cutin_transition_width, 0.8, 1.8))
        transition_end = cut_in_wind + cutin_transition_width

        # 绾︽潫1锛氬垏鍏ラ閫熶互涓?- 鍔熺巼寮哄埗涓?锛堣繖鏄墿鐞嗚寰嬶級
        below_cut_in = binned_wind < cut_in_wind
        binned_power[below_cut_in] = 0.0

        # 瀵硅繃娓″尯闂寸殑閲囨牱鐐硅繘琛屽钩婊戝鐞?
        in_transition = (binned_wind >= transition_start) & (binned_wind <= transition_end)

        if np.sum(in_transition) > 0:
            # Use points after transition as references.
            after_transition = binned_wind > transition_end
            if np.sum(after_transition) > 0:
                # Estimate end power/slope using two post-transition windows.
                win1_end = transition_end + max(0.35, 0.45 * cutin_transition_width)
                win2_end = transition_end + max(0.70, 0.90 * cutin_transition_width)
                win1 = (binned_wind > transition_end) & (binned_wind <= win1_end)
                win2 = (binned_wind > win1_end) & (binned_wind <= win2_end)

                if np.sum(win1) > 0:
                    target_power = float(np.median(binned_power[win1]))
                else:
                    reference_idx = np.where(after_transition)[0][0]
                    target_power = float(binned_power[reference_idx])

                if np.sum(win2) > 0:
                    target_power_next = float(np.median(binned_power[win2]))
                else:
                    target_power_next = target_power

                slope_end = max(10.0, (target_power_next - target_power) / max(win2_end - win1_end, 1e-6))
                transition_width = max(transition_end - cut_in_wind, 1e-6)

                # Apply Hermite transition and enforce non-decreasing slope.
                trans_idx = np.where(in_transition)[0]
                trans_idx = trans_idx[np.argsort(binned_wind[trans_idx])]

                for i in trans_idx:
                    v = binned_wind[i]
                    t = np.clip((v - cut_in_wind) / transition_width, 0.0, 1.0)

                    h00 = 2 * t**3 - 3 * t**2 + 1
                    h10 = t**3 - 2 * t**2 + t
                    h01 = -2 * t**3 + 3 * t**2
                    h11 = t**3 - t**2

                    y0 = 0.0
                    y1 = max(target_power, 0.0)
                    mean_slope = max(y1 / transition_width, 1.0)
                    m0 = np.clip(1.35 * mean_slope, 0.9 * mean_slope, 2.0 * mean_slope)
                    m1 = np.clip(slope_end, 0.45 * m0, 0.80 * m0)
                    binned_power[i] = (
                        h00 * y0 +
                        h10 * (m0 * transition_width) +
                        h01 * y1 +
                        h11 * (m1 * transition_width)
                    )

                logging.info(
                    f"  骞虫粦杩囨浮搴旂敤: {np.sum(in_transition)} 涓偣鍦ㄥ垏鍏ラ閫熷悗骞虫粦鍖?"
                    f"(width={cutin_transition_width:.2f} m/s)"
                )

        # 绾︽潫2锛氬钩鍙扮害鏉?- 楂橀閫熷尯鍩熷姛鐜囨亽瀹氾紙杩欐槸鐗╃悊涓€鑷存€х殑鍏抽敭锛?
        if 'rated_power' in self.detected_params:
            above_plateau = binned_wind >= rated_wind

            if np.sum(above_plateau) > 0:
                rated_power = float(self.detected_params.get('rated_power', 0.0))

                # Robust plateau estimate from high-wind raw points to avoid collapse
                # when aggregated plateau bins contain curtailed/negative values.
                raw_high_wind = powers[wind_speeds >= rated_wind]
                raw_high_positive = raw_high_wind[raw_high_wind > 0]

                if raw_high_positive.size > 20:
                    # Use upper-half robust center to resist curtailment-heavy tails.
                    plateau_candidates = raw_high_positive[
                        raw_high_positive > np.percentile(raw_high_positive, 40)
                    ]
                    if plateau_candidates.size > 10:
                        plateau_power = float(np.median(plateau_candidates))
                    else:
                        plateau_power = float(np.median(raw_high_positive))
                else:
                    plateau_power = rated_power

                # Keep plateau power physically near detected rated power.
                lower = max(0.0, rated_power * 0.80)
                upper = max(lower, rated_power * 1.02)
                plateau_power = float(np.clip(plateau_power, lower, upper))

                # Smoothly blend into plateau before rated_wind to avoid visible kink.
                shoulder_width = self._estimate_plateau_shoulder_width(
                    binned_wind, binned_power, rated_wind, plateau_power
                )
                self.detected_params['plateau_shoulder_width'] = shoulder_width
                shoulder_mask = (
                    (binned_wind >= rated_wind - shoulder_width) &
                    (binned_wind < rated_wind)
                )
                if np.sum(shoulder_mask) > 0:
                    t = (binned_wind[shoulder_mask] - (rated_wind - shoulder_width)) / shoulder_width
                    s = t ** 3 * (t * (t * 6 - 15) + 10)  # smootherstep (C2)
                    binned_power[shoulder_mask] = (
                        binned_power[shoulder_mask] * (1 - s) + plateau_power * s
                    )

                # Plateau region: enforce stable flat top.
                binned_power[above_plateau] = plateau_power

                logging.info(
                    f"  骞冲彴绾︽潫搴旂敤: {np.sum(above_plateau)} 涓噰鏍风偣璁句负 "
                    f"{plateau_power:.1f} kW (rated={rated_power:.1f} kW, shoulder={shoulder_width:.2f}m/s)"
                )

        return binned_wind, binned_power

    def _apply_physical_constraints(self, wind_speeds, powers):
        """
        搴旂敤鐗╃悊绾︽潫锛堣瀺鍏ヨ缁冭繃绋嬶紝闈炰簨鍚庝慨琛ワ級

        绾︽潫1锛氬崟璋冩€?- 鍒囧叆鍚庡姛鐜囦笉涓嬮檷
        绾︽潫2锛氭湁鐣屾€?- 0 鈮?P 鈮?Prated
        绾︽潫3锛氬钩鍙版湡 - 楂橀閫熷尯鍔熺巼鎭掑畾

        鏂规硶锛氱瓑娓楀洖褰?+ 鐗╃悊鎺ㄦ柇
        """
        powers_constrained = powers.copy()

        # 鑾峰彇鑷€傚簲妫€娴嬬殑鍏抽敭椋庨€?
        cut_in_wind = self.detected_params.get('cut_in_wind', 2.5)
        rated_wind = self.detected_params.get('rated_wind', 12.0)

        # 1. 鍗曡皟鎬х害鏉燂紙鍒囧叆鍒伴瀹氶閫燂級
        # Smooth then project monotonically to avoid piecewise-flat artifacts.
        monotonic_mask = (wind_speeds >= cut_in_wind) & (wind_speeds <= rated_wind)

        if np.sum(monotonic_mask) > 5:
            try:
                mono_w = wind_speeds[monotonic_mask]
                mono_p = powers_constrained[monotonic_mask]

                order = np.argsort(mono_w)
                mono_w_sorted = mono_w[order]
                mono_p_sorted = mono_p[order]

                # Light smoothing before monotonic projection.
                mono_smoothed = mono_p_sorted.copy()
                window = min(21, mono_smoothed.size if mono_smoothed.size % 2 == 1 else mono_smoothed.size - 1)
                if window >= 5:
                    mono_smoothed = savgol_filter(mono_smoothed, window_length=window, polyorder=2, mode='interp')

                # 鍗曡皟鎶曞奖
                mono_projected = np.maximum.accumulate(mono_smoothed)

                # Add a lower-bound slope in early rise to avoid 4-5 m/s stalls.
                rated_power = float(self.detected_params.get('rated_power', np.nanmax(mono_projected)))
                global_rise = max(rated_wind - cut_in_wind, 1.0)
                slope_floor = max(5.0, 0.08 * rated_power / global_rise)  # kW per m/s

                early_end = min(rated_wind - 1.0, cut_in_wind + 2.0)
                early_idx = np.where(
                    (mono_w_sorted >= cut_in_wind) & (mono_w_sorted <= early_end)
                )[0]
                for k in range(1, len(early_idx)):
                    i_prev = early_idx[k - 1]
                    i_cur = early_idx[k]
                    dv = max(mono_w_sorted[i_cur] - mono_w_sorted[i_prev], 1e-6)
                    min_allowed = mono_projected[i_prev] + slope_floor * dv
                    if mono_projected[i_cur] < min_allowed:
                        mono_projected[i_cur] = min_allowed

                # 鍐嶆淇濊瘉鍏ㄦ鍗曡皟
                mono_projected = np.maximum.accumulate(mono_projected)

                # 鍏ㄥ眬鏂滅巼姝ｅ垯鍖栵細骞虫粦涓€闃跺鏁板苟閲嶅缓锛屽噺灏戜腑娈垫姌鐐规姈鍔ㄣ€?
                if mono_projected.size >= 15:
                    local_slopes = np.diff(mono_projected) / np.maximum(np.diff(mono_w_sorted), 1e-6)
                    local_slopes = np.maximum(local_slopes, 0.0)

                    slope_window = min(
                        41,
                        local_slopes.size if local_slopes.size % 2 == 1 else local_slopes.size - 1
                    )
                    if slope_window >= 5:
                        local_slopes_smooth = savgol_filter(
                            local_slopes, window_length=slope_window, polyorder=2, mode='interp'
                        )
                    else:
                        local_slopes_smooth = local_slopes.copy()

                    local_slopes_smooth = np.maximum(local_slopes_smooth, 0.0)

                    # 鎶戝埗鐩搁偦鏂滅巼鐨勫墽鐑堣烦鍙橈紝鎻愬崌鈥滆繛璧锋潵鐪嬧€濈殑涓濇粦鎬с€?
                    for j in range(1, len(local_slopes_smooth)):
                        prev = local_slopes_smooth[j - 1]
                        upper = prev * 1.30 + 8.0
                        lower = max(0.0, prev * 0.70 - 8.0)
                        local_slopes_smooth[j] = np.clip(local_slopes_smooth[j], lower, upper)

                    mono_reconstructed = np.empty_like(mono_projected)
                    mono_reconstructed[0] = mono_projected[0]
                    for j in range(1, len(mono_projected)):
                        dv = max(mono_w_sorted[j] - mono_w_sorted[j - 1], 1e-6)
                        mono_reconstructed[j] = mono_reconstructed[j - 1] + local_slopes_smooth[j - 1] * dv

                    # 閿氬畾绔偣璺ㄥ害锛岄伩鍏嶆暣浣撴姮鍗?鍘嬬缉銆?
                    rec_span = max(mono_reconstructed[-1] - mono_reconstructed[0], 1e-6)
                    target_span = max(mono_projected[-1] - mono_projected[0], 0.0)
                    mono_reconstructed = (
                        mono_reconstructed[0] +
                        (mono_reconstructed - mono_reconstructed[0]) * (target_span / rec_span)
                    )

                    # Blend reconstructed trend with original monotone shape.
                    mono_projected = 0.72 * mono_reconstructed + 0.28 * mono_projected
                    mono_projected = np.maximum.accumulate(mono_projected)

                # 鏄犲皠鍥炲師椤哄簭
                mono_back = np.empty_like(mono_projected)
                mono_back[order] = mono_projected
                powers_constrained[monotonic_mask] = mono_back
            except:
                pass

        # 2. 骞冲彴绾︽潫锛堥珮椋庨€熷尯锛?
        plateau_mask = wind_speeds >= rated_wind

        if np.sum(plateau_mask) > 0 and 'rated_power' in self.detected_params:
            rated_power = self.detected_params['rated_power']
            # 鍏佽5%鐨勬尝鍔ㄨ寖鍥?
            powers_constrained[plateau_mask] = np.minimum(
                powers_constrained[plateau_mask],
                rated_power * 1.02
            )

        # 3. 闈炶礋绾︽潫
        powers_constrained = np.maximum(powers_constrained, 0)

        return powers_constrained

    def train(self, wind_speeds, powers, dataset_name='Unknown'):
        """
        璁粌PI虏M妯″瀷

        Args:
            wind_speeds: 璁粌椋庨€?(m/s)
            powers: 璁粌鍔熺巼 (kW)
            dataset_name: 鏁版嵁闆嗗悕绉帮紙鐢ㄤ簬鏃ュ織锛?

        Returns:
            self (璁粌濂界殑妯″瀷)
        """
        logging.info("=" * 70)
        logging.info(f"PI虏M Framework Training - Dataset: {dataset_name}")
        logging.info("=" * 70)

        # 杈撳叆楠岃瘉
        wind_speeds = np.asarray(wind_speeds).astype(float)
        powers = np.asarray(powers).astype(float)

        # 绉婚櫎鏃犳晥鍊?
        valid_mask = ~(np.isnan(wind_speeds) | np.isnan(powers) |
                      np.isinf(wind_speeds) | np.isinf(powers))
        wind_speeds = wind_speeds[valid_mask]
        powers = powers[valid_mask]

        if len(wind_speeds) < 50:
            raise ValueError(f"鏁版嵁鐐瑰お灏? {len(wind_speeds)} < 50")

        n_original = len(wind_speeds)
        logging.info(f"\n[Input] 鍘熷鏁版嵁: {n_original} 鐐?")
        logging.info(f"  椋庨€熻寖鍥? {wind_speeds.min():.2f} - {wind_speeds.max():.2f} m/s")
        logging.info(f"  鍔熺巼鑼冨洿: {powers.min():.2f} - {powers.max():.2f} kW")

        # Step 0: 妫€娴嬮瀹氬姛鐜?
        logging.info(f"\n[Step 0/5] 妫€娴嬬墿鐞嗗弬鏁?..")
        rated_power = self._detect_rated_power(wind_speeds, powers)
        logging.info(f"  棰濆畾鍔熺巼: {rated_power:.2f} kW")
        logging.info(f"  鍙惰疆鍗婂緞: {self.rotor_radius:.1f} m")

        # Step 1: 璁＄畻淇℃伅瀵嗗害锛堝畾鐞?鍜?锛?
        logging.info(f"\n[Step 1/5] 璁＄畻淇℃伅瀵嗗害...")
        info_density, v_sorted, Cp_sorted, physical_complexity, local_variance = \
            self._compute_information_density(wind_speeds, powers)

        logging.info(f"  淇℃伅瀵嗗害鑼冨洿: [{info_density.min():.3f}, {info_density.max():.3f}]")
        logging.info(f"  Cp鑼冨洿: [{Cp_sorted.min():.3f}, {Cp_sorted.max():.3f}]")
        logging.info("  Sampling density source: local variance only (Physics-InformedLoss removed)")

        # Step 2: 鑷€傚簲鍒嗙锛堝畾鐞?鐨勫簲鐢級
        logging.info(f"\n[Step 2/5] 鏋勫缓鑷€傚簲鍒嗙...")
        bin_edges = self._adaptive_binning(wind_speeds, info_density)

        logging.info(f"  鐢熸垚 {len(bin_edges)-1} 涓嚜閫傚簲绠?")
        bin_sizes = np.diff(bin_edges)
        logging.info(f"  绠卞ぇ灏忚寖鍥? [{bin_sizes.min():.2f}, {bin_sizes.max():.2f}] m/s")
        logging.info(f"  骞冲潎绠卞ぇ灏? {bin_sizes.mean():.2f} m/s")

        # Step 3: 椴佹鑱氬悎锛堝畾鐞?鐨勫簲鐢級
        logging.info(f"\n[Step 3/5] 椴佹鑱氬悎 (Huber M-estimator)...")
        binned_wind, binned_power = self._robust_aggregation_m_estimator(
            wind_speeds, powers, bin_edges
        )

        logging.info(f"  閲囨牱鐐规暟: {len(binned_wind)} (鍘熷: {n_original})")
        logging.info(f"  鍘嬬缉姣? {len(binned_wind)/n_original*100:.2f}%")
        logging.info(f"  Huber 未: {self.huber_delta}")

        # Step 4: 搴旂敤鐗╃悊绾︽潫
        logging.info(f"\n[Step 4/5] 搴旂敤鐗╃悊绾︽潫...")
        binned_power_constrained = self._apply_physical_constraints(
            binned_wind, binned_power.copy()
        )

        # Step 5: 鎷熷悎鏍锋潯
        logging.info(f"\n[Step 5/5] 鎷熷悎鐗╃悊绾︽潫鏍锋潯...")

        # 纭繚椋庨€熷崟璋?
        sort_idx = np.argsort(binned_wind)
        binned_wind_sorted = binned_wind[sort_idx]
        binned_power_sorted = binned_power_constrained[sort_idx]

        # 妫€鏌ユ槸鍚︽湁閲嶅椋庨€?
        unique_winds, unique_indices = np.unique(
            binned_wind_sorted,
            return_index=True
        )
        binned_wind_unique = binned_wind_sorted[unique_indices]
        binned_power_unique = binned_power_sorted[unique_indices]

        # 绠€鍖栵細鐩存帴鐢≒CHIP鎷熷悎锛堜繚璇佸崟璋冩€т笖淇濈暀鍘熷€硷級
        # PCHIP鏄伐绋嬪弸濂界殑鎻掑€兼柟娉曪紝鏃犻渶棰濆骞虫粦鍙傛暟
        self.spline_model = PchipInterpolator(
            binned_wind_unique,
            binned_power_unique,
            extrapolate=True
        )

        # 淇濆瓨璁粌淇℃伅
        self.training_info = {
            'n_original': n_original,
            'n_sampled': len(binned_wind_unique),
            'compression_ratio': len(binned_wind_unique) / n_original,
            'bin_edges': bin_edges,
            'binned_wind': binned_wind_unique,
            'binned_power': binned_power_unique,
            'binned_power_before_constraint': binned_power[sort_idx][unique_indices],
            'cp_curve': Cp_sorted,
            'information_density': info_density,
            'physical_complexity': physical_complexity,
            'local_variance': local_variance,
            'rated_power': rated_power
        }

        logging.info(f"  鏍锋潯鑺傜偣鏁? {len(binned_wind_unique)}")
        logging.info(f"  鎻掑€兼柟娉? PCHIP (淇濆崟璋?")

        logging.info(f"\n" + "=" * 70)
        logging.info(f"璁粌瀹屾垚!")
        logging.info(f"  鍘熷鏁版嵁: {n_original} 鐐?")
        logging.info(f"  鏍稿績閲囨牱: {len(binned_wind_unique)} 鐐?")
        logging.info(f"  鍘嬬缉姣? {len(binned_wind_unique)/n_original*100:.1f}%")
        logging.info(f"  棰濆畾鍔熺巼: {rated_power:.1f} kW")
        logging.info(f"=" * 70)

        return self

    def predict(self, wind_speeds):
        """
        棰勬祴鍔熺巼锛堝簲鐢ㄥ钩婊戣繃娓″拰鐗╃悊绾︽潫锛?

        Args:
            wind_speeds: 椋庨€?(m/s)

        Returns:
            鍔熺巼棰勬祴 (kW)
        """
        if self.spline_model is None:
            raise RuntimeError("妯″瀷鏈缁冿紝璇峰厛璋冪敤train()")

        wind_speeds = np.asarray(wind_speeds).astype(float)

        # 鏍锋潯棰勬祴
        predictions = self.spline_model(wind_speeds)

        # === 搴旂敤骞虫粦杩囨浮鍜岀墿鐞嗙害鏉?===

        # 鑾峰彇鑷€傚簲妫€娴嬬殑鍏抽敭椋庨€?
        cut_in_wind = self.detected_params.get('cut_in_wind', 3.0)
        rated_wind = self.detected_params.get('rated_wind', 12.0)

        # 绾︽潫1锛氬垏鍏ラ閫熶互涓?- 鍔熺巼寮哄埗涓?
        below_cut_in = wind_speeds < cut_in_wind
        predictions[below_cut_in] = 0.0

        # 绾︽潫2锛氶珮椋庨€熷钩鍙扮害鏉燂紙骞虫粦骞跺叆棰濆畾骞冲彴锛?
        if 'rated_power' in self.detected_params:
            rated_power = float(self.detected_params['rated_power'])
            shoulder_width = float(self.detected_params.get('plateau_shoulder_width', 0.8))
            shoulder_width = float(np.clip(shoulder_width, 0.8, 1.8))
            shoulder_start = rated_wind - shoulder_width
            shoulder_end = rated_wind + shoulder_width

            shoulder_mask = (wind_speeds >= shoulder_start) & (wind_speeds < shoulder_end)
            above_rated = wind_speeds >= shoulder_end

            if np.any(shoulder_mask):
                t = (wind_speeds[shoulder_mask] - shoulder_start) / (shoulder_end - shoulder_start)
                s = t ** 3 * (t * (t * 6 - 15) + 10)  # smootherstep (C2)
                predictions[shoulder_mask] = predictions[shoulder_mask] * (1 - s) + rated_power * s

            if np.any(above_rated):
                predictions[above_rated] = rated_power

        # Final non-negative constraint.
        predictions = np.maximum(predictions, 0.0)

        return predictions

    def get_theoretical_summary(self):
        """
        鑾峰彇鐞嗚鎬荤粨锛堢敤浜庤鏂囨挵鍐欙級
        """
        if not self.training_info:
            return "妯″瀷鏈缁?"

        return {
            'theoretical_foundation': {
                'theorem1': '涓綅鏁拌仛鍚堝湪Laplacian鍣０涓嬫渶澶у寲浜掍俊鎭?',
                'theorem2': '鏇茬巼椹卞姩閲囨牱鍦↙鈭炶寖鏁颁笅鏄渶浼樼殑',
                'integration': '鐗╃悊寮曞 + 淇℃伅鏈€澶у寲 = 椴佹閲囨牱'
            },
            'algorithm_components': {
                'adaptive_sampling': '鍩轰簬Cp(位)鏇茬巼鐨勫彉瀵嗗害鍒嗙',
                'robust_aggregation': f'Huber M-estimator (未={self.huber_delta})',
                'physical_constraints': '鍗曡皟鎬?+ 鏈夌晫鎬?+ 骞冲彴绾︽潫',
                'compression_ratio': f'{self.training_info["compression_ratio"]*100:.1f}%'
            },
            'novelty_vs_baseline': {
                'vs_uniform_binning': '鑷€傚簲 vs 鍥哄畾0.5m/s鍒嗙',
                'vs_mean_aggregation': 'Huber椴佹浼拌 vs 绠€鍗曞潎鍊?',
                'vs_post_hoc_constraints': '铻嶅叆璁粌 vs 浜嬪悗淇ˉ'
            },
            'detected_parameters': {
                'rated_power_kw': self.detected_params.get('rated_power'),
                'rotor_radius_m': self.rotor_radius
            }
        }

    def get_training_details(self):
        """鑾峰彇璇︾粏璁粌淇℃伅"""
        return self.training_info

    def evaluate(self, wind_speeds, powers_true):
        """
        璇勪及妯″瀷鎬ц兘

        Args:
            wind_speeds: 娴嬭瘯椋庨€?
            powers_true: 鐪熷疄鍔熺巼

        Returns:
            璇勪及鎸囨爣瀛楀吀
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        y_pred = self.predict(wind_speeds)

        rmse = np.sqrt(mean_squared_error(powers_true, y_pred))
        r2 = r2_score(powers_true, y_pred)
        mae = mean_absolute_error(powers_true, y_pred)

        # MAPE锛堥伩鍏嶉櫎闆讹級
        mape = np.mean(np.abs((powers_true - y_pred) / (powers_true + 1e-8))) * 100

        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'n_test': len(powers_true)
        }


class BaselineSpline:
    """
    鍩虹嚎鏍锋潯妯″瀷锛堝浐瀹?.5m/s鍒嗙 + 鍧囧€艰仛鍚堬級
    鐢ㄤ簬瀵规瘮瀹為獙
    """

    def __init__(self):
        self.spline_model = None
        self.binned_wind = None
        self.binned_power = None

    def train(self, wind_speeds, powers):
        """璁粌鍩虹嚎妯″瀷"""
        wind_speeds = np.asarray(wind_speeds)
        powers = np.asarray(powers)

        # 鍥哄畾0.5m/s鍒嗙
        bin_edges = np.arange(
            wind_speeds.min(),
            wind_speeds.max() + 0.5,
            0.5
        )

        binned_wind = []
        binned_power = []

        for i in range(len(bin_edges) - 1):
            mask = (wind_speeds >= bin_edges[i]) & (wind_speeds < bin_edges[i + 1])
            if np.any(mask):
                binned_wind.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                binned_power.append(np.mean(powers[mask]))  # 鍧囧€艰仛鍚?

        self.binned_wind = np.array(binned_wind)
        self.binned_power = np.array(binned_power)

        # 鎷熷悎鏍锋潯
        smoothing_factor = len(self.binned_wind) * 0.5
        self.spline_model = UnivariateSpline(
            self.binned_wind,
            self.binned_power,
            s=smoothing_factor,
            k=3
        )

        return self

    def predict(self, wind_speeds):
        """棰勬祴"""
        if self.spline_model is None:
            raise RuntimeError("妯″瀷鏈缁?")
        return np.maximum(0, self.spline_model(wind_speeds))


# 渚挎嵎鍑芥暟
def create_pi2m_model(**kwargs):
    """鍒涘缓PI虏M妯″瀷"""
    return PI2MFramework(**kwargs)


def create_baseline_model():
    """鍒涘缓鍩虹嚎妯″瀷"""
    return BaselineSpline()


RSBModel = PI2MFramework
create_rsb_model = create_pi2m_model

__all__ = [
    "PI2MFramework",
    "RSBModel",
    "BaselineSpline",
    "create_pi2m_model",
    "create_rsb_model",
    "create_baseline_model",
]
