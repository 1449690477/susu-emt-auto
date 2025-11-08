# -*- coding: utf-8 -*-
"""
苏苏多功能自动化工具
- Tab1：赛琪大烟花（武器突破材料本 60 级）
- Tab2：探险无尽血清 - 人物碎片自动刷取
"""

import os
import sys
import json
import time
import threading
import traceback
import copy
import queue
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ---------- 路径 ----------
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
    DATA_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = APP_DIR

BASE_DIR = DATA_DIR
TEMPLATE_DIR = os.path.join(DATA_DIR, "templates")
SCRIPTS_DIR = os.path.join(DATA_DIR, "scripts")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
SP_DIR = os.path.join(DATA_DIR, "SP")
UID_DIR = os.path.join(DATA_DIR, "UID")

MOD_DIR = os.path.join(DATA_DIR, "mod")

# 新项目：人物密函图片 / 掉落物图片
TEMPLATE_LETTERS_DIR = os.path.join(DATA_DIR, "templates_letters")
TEMPLATE_DROPS_DIR = os.path.join(DATA_DIR, "templates_drops")

for d in (
    TEMPLATE_DIR,
    SCRIPTS_DIR,
    TEMPLATE_LETTERS_DIR,
    TEMPLATE_DROPS_DIR,
    SP_DIR,
    UID_DIR,
    MOD_DIR,
):
    os.makedirs(d, exist_ok=True)

# ---------- 第三方库 ----------
try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    pyautogui = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import keyboard
except Exception:
    keyboard = None

try:
    import pygetwindow as gw
except Exception:
    gw = None

# ---------- 全局 ----------
DEFAULT_CONFIG = {
    "hotkey": "1",
    "wait_seconds": 8.0,
    "macro_a_path": "",
    "macro_b_path": "",
    "auto_loop": False,
    "guard_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
    },
    "expel_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
    },
    "mod_guard_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
    },
    "mod_expel_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
    },
}

GAME_REGION = None
worker_stop = threading.Event()
round_running_lock = threading.Lock()
hotkey_handle = None

app = None             # 赛琪大烟花 GUI 实例
fragment_apps = []     # 人物碎片 GUI 实例列表
uid_mask_manager = None

tk_call_queue = queue.Queue()
ACTIVE_FRAGMENT_GUI = None


def post_to_main_thread(func, *args, **kwargs):
    if func is None:
        return
    tk_call_queue.put((func, args, kwargs))


def start_ui_dispatch_loop(root, interval_ms: int = 30):
    def _drain_queue():
        while True:
            try:
                func, args, kwargs = tk_call_queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception:
                traceback.print_exc()
        root.after(interval_ms, _drain_queue)

    _drain_queue()


def set_active_fragment_gui(gui):
    global ACTIVE_FRAGMENT_GUI
    ACTIVE_FRAGMENT_GUI = gui


def get_active_fragment_gui():
    return ACTIVE_FRAGMENT_GUI


# 人物碎片：通用按钮名（放在 templates/）
BTN_OPEN_LETTER = "选择密函.png"
BTN_CONFIRM_LETTER = "确认选择.png"
BTN_RETREAT_START = "撤退.png"
BTN_EXPEL_NEXT_WAVE = "再次进行.png"
BTN_CONTINUE_CHALLENGE = "继续挑战.png"

AUTO_REVIVE_TEMPLATE = "x.png"
AUTO_REVIVE_THRESHOLD = 0.8
AUTO_REVIVE_CHECK_INTERVAL = 10.0
AUTO_REVIVE_HOLD_SECONDS = 6.0

UID_MASK_ALPHA = 0.92
UID_MASK_CELL = 10
UID_MASK_COLORS = ("#2e2f3a", "#4a4c5e", "#5c6075", "#3c3e4e")
UID_FIXED_MASKS = (
    # (relative_x, relative_y, width, height)
    (830, 1090, 260, 24),   # HUD 底部 UID
    (60, 1090, 260, 24),    # 左下角载入界面 UID，保持与 HUD 一致
)
UID_WINDOW_MISS_LIMIT = 60


def get_template_name(name: str, default: str) -> str:
    """Gracefully fall back to default if a global template name is missing."""
    return globals().get(name, default)


# ---------- 小工具 ----------
def format_hms(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- 日志 / 进度 ----------
def register_fragment_app(gui):
    if gui not in fragment_apps:
        fragment_apps.append(gui)


def log(msg: str):
    ts = time.strftime("[%H:%M:%S] ")
    print(ts + msg)
    if app is not None:
        app.log(msg)
    for gui in fragment_apps:
        try:
            gui.log(msg)
        except Exception:
            pass


def report_progress(p: float):
    if app is not None:
        app.set_progress(p)


GOAL_STYLE_INITIALIZED = False


def ensure_goal_progress_style():
    global GOAL_STYLE_INITIALIZED
    if GOAL_STYLE_INITIALIZED:
        return
    try:
        style = ttk.Style()
        style.configure(
            "Goal.Horizontal.TProgressbar",
            troughcolor="#ffe6fa",
            background="#5aa9ff",
            bordercolor="#f8b5dd",
            lightcolor="#8fc5ff",
            darkcolor="#ff92cf",
        )
        GOAL_STYLE_INITIALIZED = True
    except Exception:
        pass


def load_preview_image(path: str, max_size: int = 72):
    if not path or not os.path.exists(path):
        return None
    try:
        img = tk.PhotoImage(file=path)
        w = max(img.width(), 1)
        h = max(img.height(), 1)
        scale = max(1, (max(w, h) + max_size - 1) // max_size)
        if scale > 1:
            img = img.subsample(scale, scale)
        return img
    except Exception:
        return None


# ---------- 配置 ----------
def load_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            log(f"读取配置失败：{e}")
    return cfg


def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        log("配置已保存。")
    except Exception as e:
        log(f"保存配置失败：{e}")


# ---------- 游戏窗口 / 截图 ----------
def find_game_window():
    if gw is None:
        log("未安装 pygetwindow，无法定位游戏窗口。")
        return None
    try:
        wins = gw.getAllWindows()
    except Exception as e:
        log(f"获取窗口列表失败：{e}")
        return None
    for w in wins:
        title = (w.title or "")
        if "二重螺旋" in title and w.width > 400 and w.height > 300:
            return w
    log("未找到标题包含『二重螺旋』的窗口。")
    return None


def init_game_region():
    """以窗口中心 1920x1080 作为识别区域"""
    global GAME_REGION
    if pyautogui is None:
        log("未安装 pyautogui，无法截图。")
        return False
    win = find_game_window()
    if not win:
        return False
    cx = win.left + win.width // 2
    cy = win.top + win.height // 2
    GAME_REGION = (cx - 960, cy - 540, 1920, 1080)
    log(
        f"使用窗口中心区域：left={GAME_REGION[0]}, "
        f"top={GAME_REGION[1]}, w={GAME_REGION[2]}, h={GAME_REGION[3]}"
    )
    return True


def screenshot_game():
    if GAME_REGION is None:
        raise RuntimeError("GAME_REGION 未初始化")
    if pyautogui is None:
        raise RuntimeError("未安装 pyautogui")
    img = pyautogui.screenshot(region=GAME_REGION)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------- 模板匹配（templates/） ----------
def load_template(name: str):
    if cv2 is None or np is None:
        log("缺少 opencv/numpy，无法图像识别。")
        return None
    path = os.path.join(TEMPLATE_DIR, name)
    if not os.path.exists(path):
        log(f"模板不存在：{path}")
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"无法读取模板：{path}")
    return img


def match_template(name: str):
    tpl = load_template(name)
    if tpl is None:
        return 0.0, None, None
    img = screenshot_game()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    x = GAME_REGION[0] + max_loc[0] + tw // 2
    y = GAME_REGION[1] + max_loc[1] + th // 2
    return max_val, x, y


def wait_for_template(name, step_name, timeout=20.0, threshold=0.5):
    start = time.time()
    while time.time() - start < timeout and not worker_stop.is_set():
        score, _, _ = match_template(name)
        log(f"{step_name} 匹配度 {score:.3f}")
        if score >= threshold:
            log(f"{step_name} 匹配成功。")
            return True
        time.sleep(0.5)
    return False


def wait_and_click_template(name, step_name, timeout=15.0, threshold=0.8):
    start = time.time()
    while time.time() - start < timeout and not worker_stop.is_set():
        score, x, y = match_template(name)
        log(f"{step_name} 匹配度 {score:.3f}")
        if score >= threshold and x is not None:
            pyautogui.click(x, y)
            log(f"{step_name} 点击 ({x},{y})")
            return True
        time.sleep(0.5)
    return False


def click_template(name, step_name, threshold=0.7):
    score, x, y = match_template(name)
    if score >= threshold and x is not None:
        pyautogui.click(x, y)
        log(f"{step_name} 点击 ({x},{y}) 匹配度 {score:.3f}")
        return True
    log(f"{step_name} 匹配度 {score:.3f}，未点击。")
    return False


def is_exit_ui_visible(threshold=0.8) -> bool:
    """检测退图界面（exit_step1/exit_step2 任一）"""
    for nm in ("exit_step1.png", "exit_step2.png"):
        score, _, _ = match_template(nm)
        if score >= threshold:
            log(f"检测到退图界面：{nm} 匹配度 {score:.3f}")
            return True
    return False


# ---------- 模板匹配（任意路径：人物密函 / 掉落物） ----------
def load_template_from_path(path: str):
    if cv2 is None or np is None:
        log("缺少 opencv/numpy，无法图像识别。")
        return None
    if not os.path.exists(path):
        log(f"模板不存在：{path}")
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"无法读取模板：{path}")
    return img


def match_template_from_path(path: str):
    tpl = load_template_from_path(path)
    if tpl is None:
        return 0.0, None, None
    img = screenshot_game()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    x = GAME_REGION[0] + max_loc[0] + tw // 2
    y = GAME_REGION[1] + max_loc[1] + th // 2
    return max_val, x, y


def wait_and_click_template_from_path(path: str, step_name: str,
                                      timeout: float = 15.0,
                                      threshold: float = 0.8) -> bool:
    start = time.time()
    while time.time() - start < timeout and not worker_stop.is_set():
        score, x, y = match_template_from_path(path)
        log(f"{step_name} 匹配度 {score:.3f}")
        if score >= threshold and x is not None:
            pyautogui.click(x, y)
            log(f"{step_name} 点击 ({x},{y})")
            return True
        time.sleep(0.5)
    return False


class UIDMaskManager:
    """Manage UID mosaic overlays that follow the game window."""

    def __init__(self, root):
        self.root = root
        self.active = False
        self.overlays = []
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.mask_rects = UID_FIXED_MASKS

    def start(self):
        if self.active:
            messagebox.showinfo("UID遮挡", "UID遮挡已经开启。")
            return
        if not self.mask_rects:
            messagebox.showwarning("UID遮挡", "未配置任何遮挡区域。")
            return
        win = find_game_window()
        if win is None:
            messagebox.showwarning("UID遮挡", "未找到『二重螺旋』窗口。")
            return
        self.stop_event.clear()
        self.active = True
        self._create_overlays(win)
        self.monitor_thread = threading.Thread(target=self._follow_window, daemon=True)
        self.monitor_thread.start()
        log("UID 遮挡：已开启。")

    def stop(self, manual: bool = True, silent: bool = False):
        if not self.active:
            if manual and not silent:
                messagebox.showinfo("UID遮挡", "UID遮挡当前未开启。")
            return
        self.stop_event.set()
        self._destroy_overlays()
        self.monitor_thread = None
        self.active = False
        if not silent:
            log(f"UID 遮挡：{'手动' if manual else '自动'}关闭。")

    def _create_overlays(self, win):
        self._destroy_overlays()
        for idx, rect in enumerate(self.mask_rects):
            rel_x, rel_y, width, height = rect
            left = int(win.left + rel_x)
            top = int(win.top + rel_y)
            overlay = tk.Toplevel(self.root)
            overlay.withdraw()
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.attributes("-alpha", UID_MASK_ALPHA)
            base_color = UID_MASK_COLORS[idx % len(UID_MASK_COLORS)]
            overlay.configure(bg=base_color)
            canvas = tk.Canvas(
                overlay,
                width=width,
                height=height,
                highlightthickness=0,
                bd=0,
                bg=base_color,
            )
            canvas.pack(fill="both", expand=True)
            self._draw_mosaic(canvas, idx, width, height)
            overlay.geometry(f"{width}x{height}+{left}+{top}")
            overlay.deiconify()
            data = {
                "window": overlay,
                "offset_x": rel_x,
                "offset_y": rel_y,
                "width": width,
                "height": height,
            }
            with self._lock:
                self.overlays.append(data)

    def _draw_mosaic(self, canvas, seed: int, width: int, height: int):
        rnd = random.Random(1000 + seed * 131)
        for x in range(0, width, UID_MASK_CELL):
            for y in range(0, height, UID_MASK_CELL):
                color = rnd.choice(UID_MASK_COLORS)
                canvas.create_rectangle(
                    x,
                    y,
                    min(x + UID_MASK_CELL, width),
                    min(y + UID_MASK_CELL, height),
                    fill=color,
                    outline=color,
                )

    def _destroy_overlays(self):
        with self._lock:
            overlays = self.overlays
            self.overlays = []
        for data in overlays:
            win = data.get("window")
            try:
                win.destroy()
            except Exception:
                pass

    def _follow_window(self):
        miss_count = 0
        while not self.stop_event.is_set():
            win = find_game_window()
            if win is None:
                miss_count += 1
                if miss_count >= UID_WINDOW_MISS_LIMIT:
                    self.stop_event.set()
                    post_to_main_thread(
                        lambda: self._handle_auto_stop("未检测到二重螺旋窗口，UID遮挡已自动关闭。")
                    )
                    break
            else:
                miss_count = 0
                left = win.left
                top = win.top
                with self._lock:
                    overlays = list(self.overlays)
                for data in overlays:
                    self._move_overlay(data, left, top)
            time.sleep(0.05)

    def _move_overlay(self, data, left: int, top: int):
        win = data.get("window")
        if win is None:
            return
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        x = int(left + data.get("offset_x", 0))
        y = int(top + data.get("offset_y", 0))
        geom = f"{width}x{height}+{x}+{y}"
        try:
            win.geometry(geom)
        except Exception:
            pass

    def _handle_auto_stop(self, message: str):
        self.stop(manual=False, silent=True)
        if message:
            messagebox.showwarning("UID遮挡", message)
# ---------- 宏回放（EMT 风格高精度） ----------
def load_actions(path: str):
    if not path or not os.path.exists(path):
        log(f"宏文件不存在：{path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log(f"加载宏失败：{e}")
        return []
    acts = data.get("actions", [])
    if not isinstance(acts, list) or not acts:
        log(f"宏文件中没有有效动作：{path}")
        return []
    acts.sort(key=lambda a: a.get("time", 0.0))
    return acts


def play_macro(path: str, label: str,
               p1: float, p2: float,
               interrupt_on_exit: bool = False):
    """
    EMT 风格高精度回放：
    - 按 actions 里的 time 字段作为绝对时间轴
    - time.perf_counter + 自旋保证时间精度
    - interrupt_on_exit=True 时，会周期性检测退图界面，发现就提前结束宏
    """
    if keyboard is None:
        log("未安装 keyboard 模块，无法回放宏。")
        return

    actions = load_actions(path)
    if not actions:
        return

    if not label:
        label = "宏"

    total_time = float(actions[-1].get("time", 0.0))
    total_actions = len(actions)
    log(f"{label}：共 {total_actions} 个动作，时长约 {total_time:.2f} 秒。")

    start_time = time.perf_counter()
    executed_count = 0
    keyboard_count = 0
    last_progress_percent = 0

    try:
        for i, action in enumerate(actions):
            if worker_stop.is_set():
                log(f"{label}：检测到停止信号，中断宏回放。")
                break

            if interrupt_on_exit and i % 5 == 0 and is_exit_ui_visible():
                log(f"{label}：检测到退图界面，提前结束宏。")
                break

            target_time = float(action.get("time", 0.0))
            elapsed = time.perf_counter() - start_time
            sleep_time = target_time - elapsed
            if sleep_time > 0:
                if sleep_time > 0.001:
                    time.sleep(max(0, sleep_time - 0.0005))
                while time.perf_counter() - start_time < target_time:
                    pass

            try:
                ttype = action.get("type", "key_down")
                key = action.get("key")
                if ttype == "key_down" and key:
                    keyboard.press(key)
                    keyboard_count += 1
                    executed_count += 1
                elif ttype == "key_up" and key:
                    keyboard.release(key)
                    executed_count += 1
            except Exception as e:
                log(f"{label}：动作 {i} 发送失败：{e}")
                continue

            local_progress = (i + 1) / total_actions
            global_p = p1 + local_progress * (p2 - p1)
            report_progress(global_p)

            percent = int(local_progress * 100)
            if percent - last_progress_percent >= 10:
                log(f"{label} 回放进度：{percent}%（键盘:{keyboard_count}）")
                last_progress_percent = percent

        actual_elapsed = time.perf_counter() - start_time
        time_diff = actual_elapsed - total_time
        accuracy = (1 - abs(time_diff) / total_time) * 100 if total_time > 0 else 100
        log(f"{label} 执行完成：")
        log(f"  预期时长：{total_time:.3f} 秒")
        log(f"  实际耗时：{actual_elapsed:.3f} 秒")
        log(f"  时间偏差：{time_diff * 1000:.1f} 毫秒")
        log(f"  时间轴还原精度：{accuracy:.2f}%")
        log(f"  执行动作：{executed_count}/{total_actions}（键盘:{keyboard_count}）")

    finally:
        pressed = set()
        for act in actions:
            if act.get("type") == "key_down":
                pressed.add(act.get("key"))
            elif act.get("type") == "key_up":
                pressed.discard(act.get("key"))
        if pressed:
            log(f"{label}：释放未松开的按键：{', '.join(k for k in pressed if k)}")
            for k in pressed:
                try:
                    if k:
                        keyboard.release(k)
                except Exception:
                    pass


# ======================================================================
#  赛琪大烟花（老项目）
# ======================================================================
def do_enter_buttons_first_round() -> bool:
    """第一轮需要 enter_step1 / enter_step2"""
    if not wait_and_click_template("enter_step1.png", "进入 步骤1", 20.0, 0.85):
        log("进入 步骤1 失败，本轮放弃。")
        return False
    if not wait_and_click_template("enter_step2.png", "进入 步骤2", 15.0, 0.85):
        log("进入 步骤2 失败，本轮放弃。")
        return False
    return True


def check_map_by_map1() -> bool:
    """只看 map1，阈值沿用 0.5"""
    if not wait_for_template("map1.png", "地图确认（map1）", 30.0, 0.5):
        log("地图匹配失败（map1 匹配度始终低于 0.5），本轮放弃。")
        return False
    return True


def do_exit_dungeon():
    wait_and_click_template("exit_step1.png", "退图 步骤1", 20.0, 0.8)
    wait_and_click_template("exit_step2.png", "退图 步骤2", 15.0, 0.8)


def emergency_recover():
    log("执行防卡死退图：ESC → G → Q → 退图")
    try:
        if keyboard is not None:
            keyboard.press_and_release("esc")
        else:
            pyautogui.press("esc")
    except Exception as e:
        log(f"发送 ESC 失败：{e}")
    time.sleep(1.0)
    click_template("G.png", "点击 G.png", 0.6)
    time.sleep(1.0)
    click_template("Q.png", "点击 Q.png", 0.6)
    time.sleep(1.0)
    do_exit_dungeon()


def run_one_round(wait_interval: float,
                  macro_a: str,
                  macro_b: str,
                  skip_enter_buttons: bool):
    log("===== 赛琪大烟花：新一轮开始 =====")
    report_progress(0.0)

    if not init_game_region():
        log("初始化游戏区域失败，本轮结束。")
        return

    if not skip_enter_buttons:
        if not do_enter_buttons_first_round():
            return

    if not check_map_by_map1():
        return

    log("地图确认成功，等待 2 秒让画面稳定…")
    t0 = time.time()
    while time.time() - t0 < 2.0 and not worker_stop.is_set():
        time.sleep(0.1)
    report_progress(0.3)

    play_macro(macro_a, "A 阶段（靠近大烟花）", 0.3, 0.6, interrupt_on_exit=True)
    if worker_stop.is_set():
        return

    if wait_interval > 0:
        log(f"等待大烟花爆炸 {wait_interval:.1f} 秒…")
        t0 = time.time()
        while time.time() - t0 < wait_interval and not worker_stop.is_set():
            time.sleep(0.1)

    play_macro(macro_b, "B 阶段（撤退）", 0.7, 0.95, interrupt_on_exit=True)
    if worker_stop.is_set():
        return

    if is_exit_ui_visible():
        log("检测到退图按钮，执行正常退图。")
        do_exit_dungeon()
    else:
        emergency_recover()

    report_progress(1.0)
    log("赛琪大烟花：本轮完成。")


def worker_loop(wait_interval: float,
                macro_a: str,
                macro_b: str,
                auto_loop: bool):
    try:
        first_round = True
        while not worker_stop.is_set():
            skip_enter = (auto_loop and not first_round)
            if skip_enter:
                log("自动循环：本轮跳过 enter_step1/2，只从地图确认(map1)开始。")
            run_one_round(wait_interval, macro_a, macro_b, skip_enter)
            first_round = False
            if worker_stop.is_set() or not auto_loop:
                break
            log("本轮结束，3 秒后继续下一轮…")
            time.sleep(3.0)
    except Exception as e:
        log(f"后台线程异常：{e}")
        traceback.print_exc()
    finally:
        report_progress(0.0)
        log("后台线程结束。")


# ---------- GUI：赛琪大烟花 ----------
class MainGUI:
    def __init__(self, root, cfg):
        self.root = root

        self.hotkey_var = tk.StringVar(value=cfg.get("hotkey", "1"))
        self.wait_var = tk.StringVar(value=str(cfg.get("wait_seconds", 8.0)))
        self.macro_a_var = tk.StringVar(value=cfg.get("macro_a_path", ""))
        self.macro_b_var = tk.StringVar(value=cfg.get("macro_b_path", ""))
        self.auto_loop_var = tk.BooleanVar(value=cfg.get("auto_loop", False))
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="热键:").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.hotkey_var, width=15).grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="录制热键", command=self.capture_hotkey).grid(row=0, column=2, padx=3)
        ttk.Button(top, text="保存配置", command=self.save_cfg).grid(row=0, column=3, padx=3)

        tk.Label(top, text="烟花等待(秒):").grid(row=1, column=0, sticky="e")
        tk.Entry(top, textvariable=self.wait_var, width=8).grid(row=1, column=1, sticky="w")
        tk.Checkbutton(top, text="自动循环", variable=self.auto_loop_var).grid(row=1, column=2, sticky="w")

        frm2 = tk.LabelFrame(self.root, text="宏设置")
        frm2.pack(fill="x", padx=10, pady=5)

        tk.Label(frm2, text="A 宏（靠近大烟花）:").grid(row=0, column=0, sticky="e")
        tk.Entry(frm2, textvariable=self.macro_a_var, width=60).grid(row=0, column=1, sticky="w")
        ttk.Button(frm2, text="浏览…", command=self.choose_a).grid(row=0, column=2, padx=3)

        tk.Label(frm2, text="B 宏（撤退 / 退图前）:").grid(row=1, column=0, sticky="e")
        tk.Entry(frm2, textvariable=self.macro_b_var, width=60).grid(row=1, column=1, sticky="w")
        ttk.Button(frm2, text="浏览…", command=self.choose_b).grid(row=1, column=2, padx=3)

        frm3 = tk.Frame(self.root)
        frm3.pack(padx=10, pady=5)

        ttk.Button(frm3, text="开始监听热键", command=self.start_listen).grid(row=0, column=0, padx=3)
        ttk.Button(frm3, text="停止", command=self.stop_listen).grid(row=0, column=1, padx=3)
        ttk.Button(frm3, text="只执行一轮", command=self.run_once).grid(row=0, column=2, padx=3)

        frm4 = tk.LabelFrame(self.root, text="日志")
        frm4.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = tk.Text(frm4, height=10)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(frm4, command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=sb.set)

        self.progress = ttk.Progressbar(
            self.root,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.progress.pack(fill="x", padx=10, pady=5)

    def log(self, msg: str):
        ts = time.strftime("[%H:%M:%S] ")
        self.log_text.insert("end", ts + msg + "\n")
        self.log_text.see("end")

    def set_progress(self, p: float):
        self.progress_var.set(max(0.0, min(1.0, p)) * 100.0)

    # 事件
    def choose_a(self):
        p = filedialog.askopenfilename(
            title="选择 A 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_a_var.set(p)

    def choose_b(self):
        p = filedialog.askopenfilename(
            title="选择 B 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_b_var.set(p)

    def capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法录制热键。")
            return
        log("请按下你想要的热键组合…")

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
                self.hotkey_var.set(hk)
                log(f"捕获热键：{hk}")
            except Exception as e:
                log(f"录制热键失败：{e}")
        threading.Thread(target=worker, daemon=True).start()

    def save_cfg(self):
        try:
            cfg = {
                "hotkey": self.hotkey_var.get().strip(),
                "wait_seconds": float(self.wait_var.get()),
                "macro_a_path": self.macro_a_var.get(),
                "macro_b_path": self.macro_b_var.get(),
                "auto_loop": self.auto_loop_var.get(),
            }
            save_config(cfg)
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败：{e}")

    def ensure_macros(self) -> bool:
        if not self.macro_a_var.get() or not self.macro_b_var.get():
            messagebox.showwarning("提示", "请同时设置 A 宏和 B 宏。")
            return False
        return True

    def start_listen(self):
        global hotkey_handle
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法使用热键监听。")
            return
        if not self.ensure_macros():
            return
        hk = self.hotkey_var.get().strip()
        if not hk:
            messagebox.showwarning("提示", "请先设置一个热键。")
            return

        worker_stop.clear()
        if hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(hotkey_handle)
            except Exception:
                pass

        def on_hotkey():
            log("检测到热键，开始执行一轮。")
            self.start_worker(self.auto_loop_var.get())

        try:
            hotkey_handle = keyboard.add_hotkey(hk, on_hotkey)
        except Exception as e:
            messagebox.showerror("错误", f"注册热键失败：{e}")
            return
        log(f"开始监听热键：{hk}")

    def stop_listen(self):
        global hotkey_handle
        worker_stop.set()
        if keyboard is not None and hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(hotkey_handle)
            except Exception:
                pass
        hotkey_handle = None
        log("已停止监听，当前轮结束后退出。")

    def start_worker(self, auto_loop: bool):
        if not self.ensure_macros():
            return
        if not round_running_lock.acquire(blocking=False):
            log("已有一轮在运行，本次忽略。")
            return
        wait_sec = float(self.wait_var.get())
        macro_a = self.macro_a_var.get()
        macro_b = self.macro_b_var.get()

        def worker():
            try:
                worker_loop(wait_sec, macro_a, macro_b, auto_loop)
            finally:
                round_running_lock.release()
        threading.Thread(target=worker, daemon=True).start()

    def run_once(self):
        self.start_worker(auto_loop=False)


# ======================================================================
#  探险无尽血清 - 人物碎片自动刷取
# ======================================================================
class FragmentFarmGUI:
    MAX_LETTERS = 20

    def __init__(self, parent, cfg):
        self.parent = parent
        self.cfg = cfg
        self.cfg_key = getattr(self, "cfg_key", "guard_settings")
        self.letter_label = getattr(self, "letter_label", "人物密函")
        self.product_label = getattr(self, "product_label", "人物碎片")
        self.product_short_label = getattr(self, "product_short_label", "碎片")
        self.entity_label = getattr(self, "entity_label", "人物")
        self.letters_dir = getattr(self, "letters_dir", TEMPLATE_LETTERS_DIR)
        self.letters_dir_hint = getattr(self, "letters_dir_hint", "templates_letters")
        self.templates_dir_hint = getattr(self, "templates_dir_hint", "templates")
        self.preview_dir_hint = getattr(self, "preview_dir_hint", "SP")
        self.log_prefix = getattr(self, "log_prefix", "[碎片]")
        guard_cfg = cfg.get(self.cfg_key, {})

        self.wave_var = tk.StringVar(value=str(guard_cfg.get("waves", 10)))
        self.timeout_var = tk.StringVar(value=str(guard_cfg.get("timeout", 160)))
        self.auto_loop_var = tk.BooleanVar(value=True)
        self.hotkey_var = tk.StringVar(value=guard_cfg.get("hotkey", ""))

        self.selected_letter_path = None
        self.macro_a_var = tk.StringVar(value="")
        self.macro_b_var = tk.StringVar(value="")
        self.hotkey_handle = None
        self._bound_hotkey_key = None
        self.hotkey_label = self.log_prefix

        self.letter_images = []
        self.letter_buttons = []

        self.fragment_count = 0
        self.fragment_count_var = tk.StringVar(value="0")
        self.stat_name_var = tk.StringVar(value="（未选择）")
        self.stat_image = None
        self.finished_waves = 0

        self.run_start_time = None
        self.is_farming = False
        self.time_str_var = tk.StringVar(value="00:00:00")
        self.rate_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/波")
        self.eff_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/小时")
        self.wave_progress_total = 0
        self.wave_progress_count = 0
        self.wave_progress_var = tk.DoubleVar(value=0.0)
        self.wave_progress_label_var = tk.StringVar(value="轮次进度：0/0")

        self._build_ui()
        self._load_letters()
        self._update_wave_progress_ui()
        self._bind_hotkey()

    # ---- UI ----
    def _build_ui(self):
        tip_top = tk.Label(
            self.parent,
            text="只能刷『探险无尽血清』，请使用高练度的大范围水母角色！",
            fg="red",
            font=("Microsoft YaHei", 10, "bold"),
        )
        tip_top.pack(fill="x", padx=10, pady=3)

        top = tk.Frame(self.parent)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="总波数:").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.wave_var, width=6).grid(row=0, column=1, sticky="w", padx=3)
        tk.Label(top, text="（默认 10 波）").grid(row=0, column=2, sticky="w")

        tk.Label(top, text="局内超时(秒):").grid(row=0, column=3, sticky="e")
        tk.Entry(top, textvariable=self.timeout_var, width=6).grid(row=0, column=4, sticky="w", padx=3)
        tk.Label(top, text="（防卡死判定）").grid(row=0, column=5, sticky="w")

        tk.Checkbutton(
            top,
            text="开启循环",
            variable=self.auto_loop_var,
        ).grid(row=0, column=6, sticky="w", padx=10)

        hotkey_frame = tk.Frame(self.parent)
        hotkey_frame.pack(fill="x", padx=10, pady=5)
        self.hotkey_label_widget = tk.Label(
            hotkey_frame, text=f"刷{self.product_short_label}热键:"
        )
        self.hotkey_label_widget.pack(side="left")
        tk.Entry(hotkey_frame, textvariable=self.hotkey_var, width=20).pack(side="left", padx=5)
        ttk.Button(hotkey_frame, text="录制热键", command=self._capture_hotkey).pack(side="left", padx=3)
        ttk.Button(hotkey_frame, text="保存设置", command=self._save_settings).pack(side="left", padx=3)

        self.frame_letters = tk.LabelFrame(
            self.parent,
            text=f"{self.letter_label}选择（来自 {self.letters_dir_hint}/）",
        )
        self.frame_letters.pack(fill="both", expand=True, padx=10, pady=5)

        self.letters_grid = tk.Frame(self.frame_letters)
        self.letters_grid.pack(fill="both", expand=True, padx=5, pady=5)

        self.selected_label_var = tk.StringVar(value=f"当前未选择{self.letter_label}")
        self.selected_label_widget = tk.Label(
            self.frame_letters, textvariable=self.selected_label_var, fg="#0080ff"
        )
        self.selected_label_widget.pack(anchor="w", padx=5, pady=3)

        frame_macros = tk.LabelFrame(self.parent, text="地图宏脚本（mapA / mapB）")
        frame_macros.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_macros, text="mapA 宏:").grid(row=0, column=0, sticky="e")
        tk.Entry(frame_macros, textvariable=self.macro_a_var, width=50).grid(row=0, column=1, sticky="w", padx=3)
        ttk.Button(frame_macros, text="浏览…", command=self._choose_macro_a).grid(row=0, column=2, padx=3)

        tk.Label(frame_macros, text="mapB 宏:").grid(row=1, column=0, sticky="e")
        tk.Entry(frame_macros, textvariable=self.macro_b_var, width=50).grid(row=1, column=1, sticky="w", padx=3)
        ttk.Button(frame_macros, text="浏览…", command=self._choose_macro_b).grid(row=1, column=2, padx=3)

        self.stats_frame = tk.LabelFrame(
            self.parent, text=f"{self.product_label}统计（实时）"
        )
        self.stats_frame.pack(fill="x", padx=10, pady=5)

        self.stat_image_label = tk.Label(self.stats_frame, width=64, height=64, relief="sunken")
        self.stat_image_label.grid(row=0, column=0, rowspan=3, padx=5, pady=5)

        self.current_entity_label = tk.Label(
            self.stats_frame, text=f"当前{self.entity_label}："
        )
        self.current_entity_label.grid(row=0, column=1, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.stat_name_var).grid(row=0, column=2, sticky="w")

        self.total_product_label = tk.Label(
            self.stats_frame, text=f"累计{self.product_label}："
        )
        self.total_product_label.grid(row=1, column=1, sticky="e")
        tk.Label(
            self.stats_frame,
            textvariable=self.fragment_count_var,
            font=("Microsoft YaHei", 12, "bold"),
            fg="#ff6600",
        ).grid(row=1, column=2, sticky="w")

        tk.Label(self.stats_frame, text="运行时间：").grid(row=0, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.time_str_var).grid(row=0, column=4, sticky="w")

        tk.Label(self.stats_frame, text="平均掉落：").grid(row=1, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.rate_str_var).grid(row=1, column=4, sticky="w")

        tk.Label(self.stats_frame, text="效率：").grid(row=2, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.eff_str_var).grid(row=2, column=4, sticky="w")

        ensure_goal_progress_style()
        progress_box = tk.LabelFrame(self.parent, text="轮次进度")
        progress_box.pack(fill="x", padx=10, pady=5)
        ttk.Progressbar(
            progress_box,
            variable=self.wave_progress_var,
            maximum=100.0,
            style="Goal.Horizontal.TProgressbar",
        ).pack(fill="x", padx=10, pady=5)
        tk.Label(progress_box, textvariable=self.wave_progress_label_var, anchor="e").pack(fill="x", padx=10, pady=(0, 5))

        ctrl = tk.Frame(self.parent)
        ctrl.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(
            ctrl, text=f"开始刷{self.product_short_label}", command=lambda: self.start_farming()
        )
        self.start_btn.pack(side="left", padx=3)
        self.stop_btn = ttk.Button(ctrl, text="停止", command=lambda: self.stop_farming())
        self.stop_btn.pack(side="left", padx=3)

        self.log_frame = tk.LabelFrame(self.parent, text=f"{self.product_label}日志")
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = tk.Text(self.log_frame, height=10)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=sb.set)

        tip_text = (
            "提示：\n"
            f"1. {self.letter_label}图片放入 {self.letters_dir_hint}/ 目录，数量不限，本界面最多显示前 20 张。\n"
            f"2. 若需要展示{self.product_label}预览，可在 {self.preview_dir_hint}/ 目录放入与{self.letter_label}同名的 1.png / 2.png 等图片。\n"
            f"3. 按钮图（继续挑战/确认选择/撤退/mapa/mapb/G/Q/exit_step1）放在 {self.templates_dir_hint}/ 目录。\n"
        )
        self.tip_label = tk.Label(
            self.parent,
            text=tip_text,
            fg="#666666",
            anchor="w",
            justify="left",
        )
        self.tip_label.pack(fill="x", padx=10, pady=(0, 8))

    # ---- 日志 ----
    def log(self, msg: str):
        ts = time.strftime("[%H:%M:%S] ")
        self.log_text.insert("end", ts + msg + "\n")
        self.log_text.see("end")

    # ---- 人物密函 ----
    def _load_letters(self):
        for b in self.letter_buttons:
            b.destroy()
        self.letter_buttons.clear()
        self.letter_images.clear()

        files = []
        for name in os.listdir(self.letters_dir):
            low = name.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                files.append(name)
        files.sort()
        files = files[: self.MAX_LETTERS]

        if not files:
            self.selected_label_var.set(
                f"当前未选择{self.letter_label}（{self.letters_dir_hint}/ 目录为空）"
            )
            return

        max_per_row = 5
        for idx, name in enumerate(files):
            full_path = os.path.join(self.letters_dir, name)
            try:
                img = tk.PhotoImage(file=full_path)
                max_side = max(img.width(), img.height())
                if max_side > 128:
                    scale = max(1, (max_side + 127) // 128)
                    img = img.subsample(scale, scale)
            except Exception:
                continue
            self.letter_images.append(img)
            r = idx // max_per_row
            c = idx % max_per_row
            btn = tk.Button(
                self.letters_grid,
                image=img,
                relief="raised",
                borderwidth=2,
                command=lambda p=full_path, b_idx=idx: self._on_letter_clicked(p, b_idx),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self.letter_buttons.append(btn)

        if self.selected_letter_path:
            try:
                base = os.path.basename(self.selected_letter_path)
                cur_idx = files.index(base)
                self._highlight_button(cur_idx)
            except ValueError:
                self.selected_letter_path = None
                self.selected_label_var.set(f"当前未选择{self.letter_label}")

    def _on_letter_clicked(self, path: str, idx: int):
        self.selected_letter_path = path
        base = os.path.basename(path)
        self.selected_label_var.set(f"当前选择{self.letter_label}：{base}")
        self._highlight_button(idx)
        self.stat_name_var.set(base)
        self.stat_image = self.letter_images[idx]
        self.stat_image_label.config(image=self.stat_image)

    def _highlight_button(self, idx: int):
        for i, btn in enumerate(self.letter_buttons):
            if i == idx:
                btn.config(relief="sunken", bg="#a0cfff")
            else:
                btn.config(relief="raised", bg="#f0f0f0")

    # ---- 热键与设置 ----
    def _capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "当前环境未安装 keyboard，无法录制热键。")
            return

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
            except Exception as e:
                log(f"{self.log_prefix} 录制热键失败：{e}")
                return
            post_to_main_thread(lambda: self._set_hotkey(hk))

        threading.Thread(target=worker, daemon=True).start()

    def _set_hotkey(self, hotkey: str):
        self.hotkey_var.set(hotkey or "")
        self._bind_hotkey(show_popup=False)

    def _release_hotkey(self):
        if self.hotkey_handle is None or keyboard is None:
            return
        try:
            keyboard.remove_hotkey(self.hotkey_handle)
        except Exception:
            pass
        self.hotkey_handle = None
        self._bound_hotkey_key = None

    def _bind_hotkey(self, show_popup: bool = True):
        if keyboard is None:
            return
        self._release_hotkey()
        key = self.hotkey_var.get().strip()
        if not key:
            return
        try:
            handle = keyboard.add_hotkey(
                key,
                self._on_hotkey_trigger,
            )
        except Exception as e:
            log(f"{self.log_prefix} 绑定热键失败：{e}")
            messagebox.showerror("错误", f"绑定热键失败：{e}")
            return

        self.hotkey_handle = handle
        self._bound_hotkey_key = key
        log(f"{self.log_prefix} 已绑定热键：{key}")

    def _on_hotkey_trigger(self):
        post_to_main_thread(self._handle_hotkey_if_active)

    def _handle_hotkey_if_active(self):
        active = get_active_fragment_gui()
        if active is not self and not self.is_farming:
            return
        self._toggle_by_hotkey()

    def _toggle_by_hotkey(self):
        if self.is_farming:
            log(f"{self.log_prefix} 热键触发：请求停止刷{self.product_short_label}。")
            self.stop_farming(from_hotkey=True)
        else:
            log(f"{self.log_prefix} 热键触发：开始刷{self.product_short_label}。")
            self.start_farming(from_hotkey=True)

    def _save_settings(self):
        try:
            waves = int(self.wave_var.get().strip())
            if waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return
        try:
            timeout = float(self.timeout_var.get().strip())
            if timeout <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return
        section = self.cfg.setdefault(self.cfg_key, {})
        section["waves"] = waves
        section["timeout"] = timeout
        section["hotkey"] = self.hotkey_var.get().strip()
        self._bind_hotkey()
        save_config(self.cfg)
        messagebox.showinfo("提示", "设置已保存。")

    # ---- 宏选择 ----
    def _choose_macro_a(self):
        p = filedialog.askopenfilename(
            title="选择 mapA 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_a_var.set(p)

    def _choose_macro_b(self):
        p = filedialog.askopenfilename(
            title="选择 mapB 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_b_var.set(p)

    # ---- 控制 ----
    def start_farming(self, from_hotkey: bool = False):
        if not self.selected_letter_path:
            messagebox.showwarning("提示", f"请先选择一个{self.letter_label}。")
            return

        try:
            total_waves = int(self.wave_var.get().strip())
            if total_waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return

        try:
            self.timeout_seconds = float(self.timeout_var.get().strip())
            if self.timeout_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return

        if not self.macro_a_var.get() or not self.macro_b_var.get():
            messagebox.showwarning("提示", "请设置 mapA 与 mapB 的宏 JSON。")
            return

        if pyautogui is None or cv2 is None or np is None:
            messagebox.showerror("错误", "缺少 pyautogui 或 opencv/numpy，无法刷碎片。")
            return
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard 模块，无法发送按键。")
            return

        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning("提示", "当前已有其它任务在运行，请先停止后再试。")
            return

        self.fragment_count = 0
        self.fragment_count_var.set("0")
        self.finished_waves = 0
        self.run_start_time = time.time()
        self.is_farming = True
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)
        self._reset_wave_progress(total_waves)

        worker_stop.clear()
        self.start_btn.config(state="disabled")

        t = threading.Thread(target=self._farm_worker, args=(total_waves,), daemon=True)
        t.start()

        if from_hotkey:
            log(f"{self.log_prefix} 热键启动刷{self.product_short_label}成功。")

    def stop_farming(self, from_hotkey: bool = False):
        worker_stop.set()
        if not from_hotkey:
            messagebox.showinfo(
                "提示",
                f"已请求停止刷{self.product_short_label}，本波结束后将自动退出。",
            )
        else:
            log(f"{self.log_prefix} 热键停止请求已发送，等待当前波结束。")

    # ---- 统计 ----
    def _add_fragments(self, delta: int):
        if delta <= 0:
            return
        self.fragment_count += delta
        val = self.fragment_count

        def _update():
            self.fragment_count_var.set(str(val))
        post_to_main_thread(_update)

    def _update_stats_ui(self):
        if self.run_start_time is None:
            elapsed = 0
        else:
            elapsed = time.time() - self.run_start_time
        self.time_str_var.set(format_hms(elapsed))
        if self.finished_waves > 0:
            rate = self.fragment_count / self.finished_waves
        else:
            rate = 0.0
        self.rate_str_var.set(f"{rate:.2f} {self.product_short_label}/波")
        if elapsed > 0:
            eff = self.fragment_count / (elapsed / 3600.0)
        else:
            eff = 0.0
        self.eff_str_var.set(f"{eff:.2f} {self.product_short_label}/小时")

    def _stats_timer(self):
        if not self.is_farming:
            return
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)

    def _reset_wave_progress(self, total_waves: int):
        self.wave_progress_total = max(0, total_waves)
        self.wave_progress_count = 0
        self._update_wave_progress_ui()

    def _increment_wave_progress(self):
        if self.wave_progress_total <= 0:
            return
        if self.wave_progress_count < self.wave_progress_total:
            self.wave_progress_count += 1
            self._update_wave_progress_ui()

    def _force_wave_progress_complete(self):
        if self.wave_progress_total <= 0:
            return
        if self.wave_progress_count != self.wave_progress_total:
            self.wave_progress_count = self.wave_progress_total
            self._update_wave_progress_ui()

    def _update_wave_progress_ui(self):
        total = max(1, self.wave_progress_total)
        if self.wave_progress_total <= 0:
            percent = 0.0
            label = "轮次进度：0/0"
        else:
            percent = (self.wave_progress_count / total) * 100.0
            remaining = max(0, self.wave_progress_total - self.wave_progress_count)
            label = f"轮次进度：{self.wave_progress_count}/{self.wave_progress_total}（剩余 {remaining}）"
        self.wave_progress_var.set(percent)
        self.wave_progress_label_var.set(label)

    # ---- 核心刷本流程 ----
    def _farm_worker(self, total_waves: int):
        try:
            log(f"===== {self.product_label}刷取 开始 =====")
            if not init_game_region():
                messagebox.showerror(
                    "错误",
                    f"未找到『二重螺旋』窗口，无法开始刷{self.product_short_label}。",
                )
                return

            first_session = True
            session_index = 0

            while not worker_stop.is_set():
                auto_loop = self.auto_loop_var.get()
                session_index += 1
                self._reset_wave_progress(total_waves)
                log(f"{self.log_prefix} === 开始第 {session_index} 趟无尽 ===")

                if first_session:
                    if not self._enter_first_wave_and_setup():
                        return
                    first_session = False
                else:
                    if not self._restart_from_lobby_after_retreat():
                        log(f"{self.log_prefix} 循环重开失败，结束刷取。")
                        break

                current_wave = 1
                need_next_session = False

                while current_wave <= total_waves and not worker_stop.is_set():
                    log(f"{self.log_prefix} 开始第 {current_wave} 波战斗挂机…")
                    result = self._battle_and_loot(max_wait=self.timeout_seconds)
                    if worker_stop.is_set():
                        break

                    if result == "timeout":
                        log(
                            f"{self.log_prefix} 第 {current_wave} 波判定卡死，执行防卡死逻辑…"
                        )
                        if not self._anti_stuck_and_reset():
                            log(f"{self.log_prefix} 防卡死失败，结束刷取。")
                            need_next_session = False
                            break
                        # 防卡死后会重新地图识别+宏，继续当前波
                        continue

                    elif result == "ok":
                        self.finished_waves += 1
                        log(f"{self.log_prefix} 第 {current_wave} 波战斗完成。")

                        if current_wave == total_waves:
                            if auto_loop:
                                self._force_wave_progress_complete()
                                log(
                                    f"{self.log_prefix} 波数已满，已开启循环，撤退并准备下一趟。"
                                )
                                self._retreat_only()
                                need_next_session = True
                                break
                            else:
                                self._force_wave_progress_complete()
                                log(
                                    f"{self.log_prefix} 波数已满，未开启循环，撤退并结束。"
                                )
                                self._retreat_only()
                                need_next_session = False
                                worker_stop.set()
                                break
                        else:
                            if not self._enter_next_wave_without_map():
                                log(f"{self.log_prefix} 进入下一波失败，结束刷取。")
                                need_next_session = False
                                worker_stop.set()
                                break
                            current_wave += 1
                            continue

                    else:
                        need_next_session = False
                        break

                if worker_stop.is_set():
                    break
                if not auto_loop or not need_next_session:
                    break

            log(f"===== {self.product_label}刷取 结束 =====")

        except Exception as e:
            log(f"{self.log_prefix} 后台线程异常：{e}")
            traceback.print_exc()
        finally:
            worker_stop.clear()
            round_running_lock.release()
            self.is_farming = False
            self._update_stats_ui()

            def restore():
                try:
                    self.start_btn.config(state="normal")
                except Exception:
                    pass
            post_to_main_thread(restore)

            if self.run_start_time is not None:
                elapsed = time.time() - self.run_start_time
                time_str = format_hms(elapsed)
                if self.finished_waves > 0:
                    rate = self.fragment_count / self.finished_waves
                else:
                    rate = 0.0
                if elapsed > 0:
                    eff = self.fragment_count / (elapsed / 3600.0)
                else:
                    eff = 0.0
                msg = (
                    f"{self.product_label}刷取已结束。\n\n"
                    f"总运行时间：{time_str}\n"
                    f"完成波数：{self.finished_waves}\n"
                    f"累计{self.product_label}：{self.fragment_count}\n"
                    f"平均掉落：{rate:.2f} {self.product_short_label}/波\n"
                    f"效率：{eff:.2f} {self.product_short_label}/小时\n"
                )
                post_to_main_thread(
                    lambda: messagebox.showinfo(
                        f"刷{self.product_short_label}完成", msg
                    )
                )

    # ---- 首次进图 / 循环重开 ----
    def _enter_first_wave_and_setup(self) -> bool:
        log(
            f"{self.log_prefix} 首次进图：选择密函按钮 → {self.letter_label} → 确认选择 → 地图AB识别 + 宏"
        )
        btn_open_letter = get_template_name("BTN_OPEN_LETTER", "选择密函.png")
        if not wait_and_click_template(
            btn_open_letter,
            f"{self.log_prefix} 首次：选择密函按钮",
            25.0,
            0.8,
        ):
            log(f"{self.log_prefix} 首次：未能点击 选择密函.png。")
            return False
        if not wait_and_click_template_from_path(
            self.selected_letter_path,
            f"{self.log_prefix} 首次：点击{self.letter_label}",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 首次：未能点击{self.letter_label}。")
            return False
        if not wait_and_click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 首次：确认选择",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 首次：未能点击 确认选择.png。")
            return False
        self._increment_wave_progress()
        return self._map_detect_and_run_macros()

    def _restart_from_lobby_after_retreat(self) -> bool:
        log(
            f"{self.log_prefix} 循环重开：再次进行 → {self.letter_label} → 确认选择 → 地图AB + 宏"
        )
        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 循环重开：再次进行按钮",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 循环重开：未能点击 再次进行.png。")
            return False
        if not wait_and_click_template_from_path(
            self.selected_letter_path,
            f"{self.log_prefix} 循环重开：点击{self.letter_label}",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 循环重开：未能点击{self.letter_label}。")
            return False
        if not wait_and_click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 循环重开：确认选择",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 循环重开：未能点击 确认选择.png。")
            return False
        self._increment_wave_progress()
        return self._map_detect_and_run_macros()

    def _map_detect_and_run_macros(self) -> bool:
        """
        确认密函后，持续匹配 mapa / mapb：
        - 最多 12 秒
        - 任意一张匹配度 >= 0.7 就认定地图
        - 然后再等待 2 秒，最后执行对应宏
        """
        log(f"{self.log_prefix} 开始持续识别地图 A/B（最长 12 秒）…")

        deadline = time.time() + 12.0
        chosen = None
        score_a = 0.0
        score_b = 0.0

        while time.time() < deadline and not worker_stop.is_set():
            score_a, _, _ = match_template("mapa.png")
            score_b, _, _ = match_template("mapb.png")
            log(
                f"{self.log_prefix} mapa 匹配度 {score_a:.3f}，mapb 匹配度 {score_b:.3f}"
            )

            best = max(score_a, score_b)
            if best >= 0.7:
                chosen = "A" if score_a >= score_b else "B"
                break

            time.sleep(0.4)

        if chosen is None:
            log(f"{self.log_prefix} 12 秒内地图匹配度始终低于 0.7，本趟放弃。")
            return False

        if chosen == "A":
            macro_path = self.macro_a_var.get()
            label = "mapA 宏"
        else:
            macro_path = self.macro_b_var.get()
            label = "mapB 宏"

        if not macro_path or not os.path.exists(macro_path):
            log(f"{self.log_prefix} {label} 文件不存在：{macro_path}")
            return False

        log(
            f"{self.log_prefix} 识别为 {label}（mapa={score_a:.3f}, mapb={score_b:.3f}），"
            "再等待 2 秒后执行宏…"
        )

        t0 = time.time()
        while time.time() - t0 < 2.0 and not worker_stop.is_set():
            time.sleep(0.1)

        play_macro(macro_path, f"{self.log_prefix} {label}", 0.0, 0.3, interrupt_on_exit=False)
        return True

    # ---- 掉落界面检测 & 掉落识别 ----
    def _is_drop_ui_visible(self, log_detail: bool = False, threshold: float = 0.7) -> bool:
        """
        判断当前是否已经进入『物品掉落选择界面』：
        用确认按钮『确认选择.png』做判定，匹配度 >= threshold 才算界面出现。
        """
        score, _, _ = match_template(BTN_CONFIRM_LETTER)
        if log_detail:
            log(f"{self.log_prefix} 掉落界面检查：确认选择 匹配度 {score:.3f}")
        return score >= threshold

    def _detect_and_pick_drop(self, threshold=0.8) -> bool:
        """
        已经确认『物品掉落界面』出现之后调用：

        现在不再识别具体掉落物，直接点击『确认选择』进入下一步。
        """
        if click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 掉落确认：确认选择",
            threshold=0.7,
        ):
            time.sleep(1.0)
            return True
        return False

    def _auto_revive_if_needed(self) -> bool:
        template_path = os.path.join(TEMPLATE_DIR, AUTO_REVIVE_TEMPLATE)
        if not os.path.exists(template_path):
            return False
        score, _, _ = match_template(AUTO_REVIVE_TEMPLATE)
        if score >= AUTO_REVIVE_THRESHOLD:
            log(
                f"{self.log_prefix} 检测到角色死亡（{AUTO_REVIVE_TEMPLATE} 匹配度 {score:.3f}），执行长按 X 复苏。"
            )
            if not self._press_and_hold_key("x", AUTO_REVIVE_HOLD_SECONDS):
                log(f"{self.log_prefix} 长按 X 失败，无法执行自动复苏。")
                return False
            log(f"{self.log_prefix} 自动复苏完成，继续战斗挂机。")
            return True
        return False

    def _press_and_hold_key(self, key: str, duration: float) -> bool:
        if keyboard is None and pyautogui is None:
            return False
        pressed = False
        try:
            if keyboard is not None:
                keyboard.press(key)
            else:
                pyautogui.keyDown(key)
            pressed = True
            time.sleep(duration)
            return True
        except Exception as e:
            log(f"{self.log_prefix} 长按 {key} 失败：{e}")
            return False
        finally:
            if pressed:
                try:
                    if keyboard is not None:
                        keyboard.release(key)
                    else:
                        pyautogui.keyUp(key)
                except Exception:
                    pass

    def _battle_and_loot(self, max_wait: float = 160.0) -> str:
        """
        战斗挂机 + 掉落判断，严格遵守 max_wait（例如 160 秒）：

        - 宏执行完之后调用本函数
        - 每 5 秒按一次 E
        - 在 [0, max_wait] 内循环：
            1) 先判断『物品掉落界面』是否出现（确认选择.png 匹配度 >= 0.7）
            2) 只有界面出现以后，才去识别掉落物并选择
        - 如果在 max_wait 秒内成功选到了掉落物 → 返回 'ok'
        - 如果超过 max_wait 仍然没检测到掉落界面/没选到 → 返回 'timeout'
        """
        if keyboard is None and pyautogui is None:
            log(f"{self.log_prefix} 无法发送按键。")
            return "stopped"

        log(
            f"{self.log_prefix} 开始战斗挂机（每 5 秒按一次 E，超时 {max_wait:.1f} 秒）。"
        )
        start = time.time()
        last_e = 0.0
        last_revive_check = start

        min_drop_check_time = 10.0
        drop_ui_visible = False
        last_ui_log = 0.0

        while not worker_stop.is_set():
            now = time.time()

            if now - last_revive_check >= AUTO_REVIVE_CHECK_INTERVAL:
                last_revive_check = now
                self._auto_revive_if_needed()

            if now - last_e >= 5.0:
                try:
                    if keyboard is not None:
                        keyboard.press_and_release("e")
                    else:
                        pyautogui.press("e")
                except Exception as e:
                    log(f"{self.log_prefix} 发送 E 失败：{e}")
                last_e = now

            if now - start >= min_drop_check_time:
                if not drop_ui_visible:
                    if self._is_drop_ui_visible():
                        drop_ui_visible = True
                        log(f"{self.log_prefix} 检测到物品掉落界面，开始识别掉落物。")
                    else:
                        if now - last_ui_log > 3.0:
                            self._is_drop_ui_visible(log_detail=True)
                            last_ui_log = now
                else:
                    if self._detect_and_pick_drop():
                        log(f"{self.log_prefix} 本波掉落已选择。")
                        return "ok"

            if now - start > max_wait:
                log(f"{self.log_prefix} 超过 {max_wait:.1f} 秒未检测到掉落，判定卡死。")
                return "timeout"

            time.sleep(0.5)

        return "stopped"

    # ---- 正常进入下一波（不做地图识别） ----
    def _enter_next_wave_without_map(self) -> bool:
        log(
            f"{self.log_prefix} 进入下一波：再次进行 → {self.letter_label} → 确认选择"
        )
        if not wait_and_click_template(
            BTN_CONTINUE_CHALLENGE,
            f"{self.log_prefix} 下一波：继续挑战按钮",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 下一波：未能点击 继续挑战.png。")
            return False
        self._increment_wave_progress()
        if not wait_and_click_template_from_path(
            self.selected_letter_path,
            f"{self.log_prefix} 下一波：点击{self.letter_label}",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 下一波：未能点击{self.letter_label}。")
            return False
        if not wait_and_click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 下一波：确认选择",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 下一波：未能点击 确认选择.png。")
            return False
        time.sleep(2.0)
        return True

    # ---- 防卡死 ----
    def _anti_stuck_and_reset(self) -> bool:
        """
        防卡死：Esc → G → Q → 再次进行 → 人物密函 → 确认 → 地图识别
        """
        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            else:
                pyautogui.press("esc")
        except Exception as e:
            log(f"{self.log_prefix} 发送 ESC 失败：{e}")
        time.sleep(1.0)
        click_template("G.png", f"{self.log_prefix} 防卡死：点击 G.png", 0.6)
        time.sleep(1.0)
        click_template("Q.png", f"{self.log_prefix} 防卡死：点击 Q.png", 0.6)
        time.sleep(1.0)

        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 防卡死：再次进行按钮",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 防卡死：未能点击 再次进行.png。")
            return False

        if not wait_and_click_template_from_path(
            self.selected_letter_path,
            f"{self.log_prefix} 防卡死：点击{self.letter_label}",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 防卡死：未能点击{self.letter_label}。")
            return False

        if not wait_and_click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 防卡死：确认选择",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 防卡死：未能点击 确认选择.png。")
            return False

        return self._map_detect_and_run_macros()

    # ---- 撤退 ----
    def _retreat_only(self):
        wait_and_click_template(
            BTN_RETREAT_START,
            f"{self.log_prefix} 撤退按钮",
            20.0,
            0.8,
        )


class ExpelFragmentGUI:
    MAX_LETTERS = 20

    def __init__(self, parent, cfg):
        self.parent = parent
        self.cfg = cfg
        self.cfg_key = getattr(self, "cfg_key", "expel_settings")
        self.letter_label = getattr(self, "letter_label", "人物密函")
        self.product_label = getattr(self, "product_label", "人物碎片")
        self.product_short_label = getattr(self, "product_short_label", "碎片")
        self.entity_label = getattr(self, "entity_label", "人物")
        self.letters_dir = getattr(self, "letters_dir", TEMPLATE_LETTERS_DIR)
        self.letters_dir_hint = getattr(self, "letters_dir_hint", "templates_letters")
        self.templates_dir_hint = getattr(self, "templates_dir_hint", "templates")
        self.preview_dir_hint = getattr(self, "preview_dir_hint", "SP")
        self.log_prefix = getattr(self, "log_prefix", "[驱离]")
        expel_cfg = cfg.get(self.cfg_key, {})

        self.wave_var = tk.StringVar(value=str(expel_cfg.get("waves", 10)))
        self.timeout_var = tk.StringVar(value=str(expel_cfg.get("timeout", 160)))
        self.auto_loop_var = tk.BooleanVar(value=True)
        self.hotkey_var = tk.StringVar(value=expel_cfg.get("hotkey", ""))

        self.selected_letter_path = None

        self.letter_images = []
        self.letter_buttons = []

        self.fragment_count = 0
        self.fragment_count_var = tk.StringVar(value="0")
        self.stat_name_var = tk.StringVar(value="（未选择）")
        self.stat_image = None
        self.finished_waves = 0

        self.run_start_time = None
        self.is_farming = False
        self.time_str_var = tk.StringVar(value="00:00:00")
        self.rate_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/波")
        self.eff_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/小时")
        self.hotkey_handle = None
        self._bound_hotkey_key = None
        self.hotkey_label = self.log_prefix

        self._build_ui()
        self._load_letters()
        self._bind_hotkey()

    def _build_ui(self):
        tip_top = tk.Label(
            self.parent,
            text=(
                f"驱离模式：选择{self.letter_label}后自动等待 7 秒进入地图 → W 键前进 10 秒 → 随机 WASD + 每 5 秒按一次 E。"
            ),
            fg="red",
            font=("Microsoft YaHei", 10, "bold"),
        )
        tip_top.pack(fill="x", padx=10, pady=3)

        top = tk.Frame(self.parent)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="总波数:").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.wave_var, width=6).grid(row=0, column=1, sticky="w", padx=3)
        tk.Label(top, text="（默认 10 波）").grid(row=0, column=2, sticky="w")

        tk.Label(top, text="局内超时(秒):").grid(row=0, column=3, sticky="e")
        tk.Entry(top, textvariable=self.timeout_var, width=6).grid(row=0, column=4, sticky="w", padx=3)
        tk.Label(top, text="（防卡死判定）").grid(row=0, column=5, sticky="w")

        tk.Checkbutton(
            top,
            text="开启循环",
            variable=self.auto_loop_var,
        ).grid(row=0, column=6, sticky="w", padx=10)

        hotkey_frame = tk.Frame(self.parent)
        hotkey_frame.pack(fill="x", padx=10, pady=5)
        self.hotkey_label_widget = tk.Label(
            hotkey_frame, text=f"刷{self.product_short_label}热键:"
        )
        self.hotkey_label_widget.pack(side="left")
        tk.Entry(hotkey_frame, textvariable=self.hotkey_var, width=20).pack(side="left", padx=5)
        ttk.Button(hotkey_frame, text="录制热键", command=self._capture_hotkey).pack(side="left", padx=3)
        ttk.Button(hotkey_frame, text="保存设置", command=self._save_settings).pack(side="left", padx=3)

        self.frame_letters = tk.LabelFrame(
            self.parent,
            text=f"{self.letter_label}选择（来自 {self.letters_dir_hint}/）",
        )
        self.frame_letters.pack(fill="both", expand=True, padx=10, pady=5)

        self.letters_grid = tk.Frame(self.frame_letters)
        self.letters_grid.pack(fill="both", expand=True, padx=5, pady=5)

        self.selected_label_var = tk.StringVar(value=f"当前未选择{self.letter_label}")
        self.selected_label_widget = tk.Label(
            self.frame_letters, textvariable=self.selected_label_var, fg="#0080ff"
        )
        self.selected_label_widget.pack(anchor="w", padx=5, pady=3)

        self.stats_frame = tk.LabelFrame(
            self.parent, text=f"{self.product_label}统计（实时）"
        )
        self.stats_frame.pack(fill="x", padx=10, pady=5)

        self.stat_image_label = tk.Label(self.stats_frame, width=64, height=64, relief="sunken")
        self.stat_image_label.grid(row=0, column=0, rowspan=3, padx=5, pady=5)

        self.current_entity_label = tk.Label(
            self.stats_frame, text=f"当前{self.entity_label}："
        )
        self.current_entity_label.grid(row=0, column=1, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.stat_name_var).grid(row=0, column=2, sticky="w")

        self.total_product_label = tk.Label(
            self.stats_frame, text=f"累计{self.product_label}："
        )
        self.total_product_label.grid(row=1, column=1, sticky="e")
        tk.Label(
            self.stats_frame,
            textvariable=self.fragment_count_var,
            font=("Microsoft YaHei", 12, "bold"),
            fg="#ff6600",
        ).grid(row=1, column=2, sticky="w")

        tk.Label(self.stats_frame, text="运行时间：").grid(row=0, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.time_str_var).grid(row=0, column=4, sticky="w")

        tk.Label(self.stats_frame, text="平均掉落：").grid(row=1, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.rate_str_var).grid(row=1, column=4, sticky="w")

        tk.Label(self.stats_frame, text="效率：").grid(row=2, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.eff_str_var).grid(row=2, column=4, sticky="w")

        ctrl = tk.Frame(self.parent)
        ctrl.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(
            ctrl, text=f"开始刷{self.product_short_label}", command=lambda: self.start_farming()
        )
        self.start_btn.pack(side="left", padx=3)
        self.stop_btn = ttk.Button(ctrl, text="停止", command=lambda: self.stop_farming())
        self.stop_btn.pack(side="left", padx=3)

        self.log_frame = tk.LabelFrame(self.parent, text="驱离模式日志")
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = tk.Text(self.log_frame, height=10)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=sb.set)

        tip_text = (
            "提示：\n"
            f"1. 本模式无需 mapA / mapB 宏，确认{self.letter_label}后默认 7 秒进入地图。\n"
            f"2. {self.letter_label}图片放入 {self.letters_dir_hint}/ 目录，常用按钮模板仍存放在 {self.templates_dir_hint}/ 目录。\n"
            "3. 若卡死会自动执行 Esc→G→Q→exit_step1 的防卡死流程，并重新开始当前波。\n"
        )
        self.tip_label = tk.Label(
            self.parent,
            text=tip_text,
            fg="#666666",
            anchor="w",
            justify="left",
        )
        self.tip_label.pack(fill="x", padx=10, pady=(0, 8))

    def log(self, msg: str):
        ts = time.strftime("[%H:%M:%S] ")
        self.log_text.insert("end", ts + msg + "\n")
        self.log_text.see("end")

    def _load_letters(self):
        for b in self.letter_buttons:
            b.destroy()
        self.letter_buttons.clear()
        self.letter_images.clear()

        files = []
        for name in os.listdir(self.letters_dir):
            low = name.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                files.append(name)
        files.sort()
        files = files[: self.MAX_LETTERS]

        if not files:
            self.selected_label_var.set(
                f"当前未选择{self.letter_label}（{self.letters_dir_hint}/ 目录为空）"
            )
            return

        max_per_row = 5
        for idx, name in enumerate(files):
            full_path = os.path.join(self.letters_dir, name)
            try:
                img = tk.PhotoImage(file=full_path)
                max_side = max(img.width(), img.height())
                if max_side > 128:
                    scale = max(1, (max_side + 127) // 128)
                    img = img.subsample(scale, scale)
            except Exception:
                continue
            self.letter_images.append(img)
            r = idx // max_per_row
            c = idx % max_per_row
            btn = tk.Button(
                self.letters_grid,
                image=img,
                relief="raised",
                borderwidth=2,
                command=lambda p=full_path, b_idx=idx: self._on_letter_clicked(p, b_idx),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self.letter_buttons.append(btn)

        if self.selected_letter_path:
            try:
                base = os.path.basename(self.selected_letter_path)
                cur_idx = files.index(base)
                self._highlight_button(cur_idx)
            except ValueError:
                self.selected_letter_path = None
                self.selected_label_var.set(f"当前未选择{self.letter_label}")

    def _on_letter_clicked(self, path: str, idx: int):
        self.selected_letter_path = path
        base = os.path.basename(path)
        self.selected_label_var.set(f"当前选择{self.letter_label}：{base}")
        self._highlight_button(idx)
        self.stat_name_var.set(base)
        self.stat_image = self.letter_images[idx]
        self.stat_image_label.config(image=self.stat_image)

    def _highlight_button(self, idx: int):
        for i, btn in enumerate(self.letter_buttons):
            if i == idx:
                btn.config(relief="sunken", bg="#a0cfff")
            else:
                btn.config(relief="raised", bg="#f0f0f0")

    # ---- 热键与设置 ----
    def _capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "当前环境未安装 keyboard，无法录制热键。")
            return

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
            except Exception as e:
                log(f"{self.log_prefix} 录制热键失败：{e}")
                return
            post_to_main_thread(lambda: self._set_hotkey(hk))

        threading.Thread(target=worker, daemon=True).start()

    def _set_hotkey(self, hotkey: str):
        self.hotkey_var.set(hotkey or "")
        self._bind_hotkey(show_popup=False)

    def _release_hotkey(self):
        if self.hotkey_handle is None or keyboard is None:
            return
        try:
            keyboard.remove_hotkey(self.hotkey_handle)
        except Exception:
            pass
        self.hotkey_handle = None
        self._bound_hotkey_key = None

    def _bind_hotkey(self, show_popup: bool = True):
        if keyboard is None:
            return
        self._release_hotkey()
        key = self.hotkey_var.get().strip()
        if not key:
            return
        try:
            handle = keyboard.add_hotkey(
                key,
                self._on_hotkey_trigger,
            )
        except Exception as e:
            log(f"{self.log_prefix} 绑定热键失败：{e}")
            messagebox.showerror("错误", f"绑定热键失败：{e}")
            return

        self.hotkey_handle = handle
        self._bound_hotkey_key = key
        log(f"{self.log_prefix} 已绑定热键：{key}")

    def _on_hotkey_trigger(self):
        post_to_main_thread(self._handle_hotkey_if_active)

    def _handle_hotkey_if_active(self):
        active = get_active_fragment_gui()
        if active is not self and not self.is_farming:
            return
        self._toggle_by_hotkey()

    def _toggle_by_hotkey(self):
        if self.is_farming:
            log(f"{self.log_prefix} 热键触发：请求停止刷{self.product_short_label}。")
            self.stop_farming(from_hotkey=True)
        else:
            log(f"{self.log_prefix} 热键触发：开始刷{self.product_short_label}。")
            self.start_farming(from_hotkey=True)

    def _save_settings(self):
        try:
            waves = int(self.wave_var.get().strip())
            if waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return
        try:
            timeout = float(self.timeout_var.get().strip())
            if timeout <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return
        section = self.cfg.setdefault(self.cfg_key, {})
        section["waves"] = waves
        section["timeout"] = timeout
        section["hotkey"] = self.hotkey_var.get().strip()
        self._bind_hotkey()
        save_config(self.cfg)
        messagebox.showinfo("提示", "设置已保存。")

    def start_farming(self, from_hotkey: bool = False):
        if not self.selected_letter_path:
            messagebox.showwarning("提示", f"请先选择一个{self.letter_label}。")
            return

        try:
            total_waves = int(self.wave_var.get().strip())
            if total_waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return

        try:
            self.timeout_seconds = float(self.timeout_var.get().strip())
            if self.timeout_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return

        if pyautogui is None or cv2 is None or np is None:
            messagebox.showerror("错误", "缺少 pyautogui 或 opencv/numpy，无法刷碎片。")
            return
        if keyboard is None and not hasattr(pyautogui, "keyDown"):
            messagebox.showerror("错误", "当前环境无法发送键盘输入。")
            return

        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning("提示", "当前已有其它任务在运行，请先停止后再试。")
            return

        self.fragment_count = 0
        self.fragment_count_var.set("0")
        self.finished_waves = 0
        self.run_start_time = time.time()
        self.is_farming = True
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)

        worker_stop.clear()
        self.start_btn.config(state="disabled")

        t = threading.Thread(target=self._expel_worker, args=(total_waves,), daemon=True)
        t.start()

        if from_hotkey:
            log(f"{self.log_prefix} 热键启动刷{self.product_short_label}成功。")

    def stop_farming(self, from_hotkey: bool = False):
        worker_stop.set()
        if not from_hotkey:
            messagebox.showinfo(
                "提示",
                f"已请求停止刷{self.product_short_label}，本波结束后将自动退出。",
            )
        else:
            log(f"{self.log_prefix} 热键停止请求已发送，等待当前波结束。")

    def _add_fragments(self, delta: int):
        if delta <= 0:
            return
        self.fragment_count += delta
        val = self.fragment_count

        def _update():
            self.fragment_count_var.set(str(val))

        post_to_main_thread(_update)

    def _update_stats_ui(self):
        if self.run_start_time is None:
            elapsed = 0
        else:
            elapsed = time.time() - self.run_start_time
        self.time_str_var.set(format_hms(elapsed))
        if self.finished_waves > 0:
            rate = self.fragment_count / self.finished_waves
        else:
            rate = 0.0
        self.rate_str_var.set(f"{rate:.2f} {self.product_short_label}/波")
        if elapsed > 0:
            eff = self.fragment_count / (elapsed / 3600.0)
        else:
            eff = 0.0
        self.eff_str_var.set(f"{eff:.2f} {self.product_short_label}/小时")

    def _stats_timer(self):
        if not self.is_farming:
            return
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)

    def _expel_worker(self, total_waves: int):
        try:
            log("===== 驱离刷取 开始 =====")
            if not init_game_region():
                messagebox.showerror("错误", "未找到『二重螺旋』窗口，无法开始驱离刷取。")
                return

            if not self._prepare_first_wave():
                log(f"{self.log_prefix} 首次进入失败，结束刷取。")
                return

            current_wave = 1
            max_wave = total_waves

            while not worker_stop.is_set():
                log(f"{self.log_prefix} 开始第 {current_wave} 波战斗挂机…")
                result = self._run_wave_actions(current_wave)
                if worker_stop.is_set():
                    break

                if result == "timeout":
                    log(f"{self.log_prefix} 第 {current_wave} 波判定卡死，执行防卡死逻辑…")
                    if not self._anti_stuck_and_reset():
                        log(f"{self.log_prefix} 防卡死失败，结束刷取。")
                        break
                    continue

                if result != "ok":
                    break

                self.finished_waves += 1

                if max_wave > 0 and current_wave >= max_wave:
                    if self.auto_loop_var.get():
                        current_wave = 1
                    else:
                        log(f"{self.log_prefix} 到达设定波数（未启用自动循环），撤退并结束。")
                        self._retreat_only()
                        break
                else:
                    current_wave += 1

                if worker_stop.is_set():
                    break
                if not self.auto_loop_var.get() and max_wave > 0 and self.finished_waves >= max_wave:
                    # 已经完成指定波数且不循环，直接退出
                    self._retreat_only()
                    break

                if not self._prepare_next_wave():
                    log(f"{self.log_prefix} 进入下一波失败，结束刷取。")
                    break

            log("===== 驱离刷取 结束 =====")

        except Exception as e:
            log(f"{self.log_prefix} 后台线程异常：{e}")
            traceback.print_exc()
        finally:
            worker_stop.clear()
            round_running_lock.release()
            self.is_farming = False
            self._update_stats_ui()

            def restore():
                try:
                    self.start_btn.config(state="normal")
                except Exception:
                    pass

            post_to_main_thread(restore)

            if self.run_start_time is not None:
                elapsed = time.time() - self.run_start_time
                time_str = format_hms(elapsed)
                if self.finished_waves > 0:
                    rate = self.fragment_count / self.finished_waves
                else:
                    rate = 0.0
                if elapsed > 0:
                    eff = self.fragment_count / (elapsed / 3600.0)
                else:
                    eff = 0.0
                msg = (
                    f"驱离刷{self.product_short_label}已结束。\n\n"
                    f"总运行时间：{time_str}\n"
                    f"完成波数：{self.finished_waves}\n"
                    f"累计{self.product_label}：{self.fragment_count}\n"
                    f"平均掉落：{rate:.2f} {self.product_short_label}/波\n"
                    f"效率：{eff:.2f} {self.product_short_label}/小时\n"
                )
                post_to_main_thread(
                    lambda: messagebox.showinfo(
                        f"驱离刷{self.product_short_label}完成", msg
                    )
                )

    def _prepare_first_wave(self) -> bool:
        log(f"{self.log_prefix} 首次进图：{self.letter_label} → 确认选择")
        return self._select_letter_sequence(f"{self.log_prefix} 首次", need_open_button=True)

    def _prepare_next_wave(self) -> bool:
        log(f"{self.log_prefix} 下一波：再次进行 → {self.letter_label} → 确认")
        if not wait_and_click_template(BTN_EXPEL_NEXT_WAVE, f"{self.log_prefix} 下一波：再次进行按钮", 25.0, 0.8):
            log(f"{self.log_prefix} 下一波：未能点击 再次进行.png。")
            return False
        return self._select_letter_sequence(f"{self.log_prefix} 下一波", need_open_button=False)

    def _select_letter_sequence(self, prefix: str, need_open_button: bool) -> bool:
        if need_open_button:
            btn_open_letter = get_template_name("BTN_OPEN_LETTER", "选择密函.png")
            if not wait_and_click_template(
                btn_open_letter,
                f"{prefix}：选择密函按钮",
                20.0,
                0.8,
            ):
                log(f"{prefix}：未能点击 选择密函.png。")
                return False

        if not wait_and_click_template_from_path(
            self.selected_letter_path,
            f"{prefix}：点击{self.letter_label}",
            20.0,
            0.8,
        ):
            log(f"{prefix}：未能点击{self.letter_label}。")
            return False
        if not wait_and_click_template(BTN_CONFIRM_LETTER, f"{prefix}：确认选择", 20.0, 0.8):
            log(f"{prefix}：未能点击 确认选择.png。")
            return False
        return True

    def _run_wave_actions(self, wave_index: int) -> str:
        if not self._wait_for_map_entry():
            return "stopped"
        if not self._hold_forward(12.0):
            return "stopped"
        return self._random_move_and_loot(self.timeout_seconds)

    def _wait_for_map_entry(self, wait_seconds: float = 7.0) -> bool:
        log(f"{self.log_prefix} 确认后等待 {wait_seconds:.1f} 秒让地图载入…")
        start = time.time()
        while time.time() - start < wait_seconds:
            if worker_stop.is_set():
                return False
            time.sleep(0.1)
        return True

    def _hold_forward(self, duration: float) -> bool:
        if keyboard is None and not hasattr(pyautogui, "keyDown"):
            log(f"{self.log_prefix} 无法发送按键，无法执行长按 W。")
            return False
        log(f"{self.log_prefix} 长按 W {duration:.1f} 秒…")
        self._press_key("w")
        try:
            start = time.time()
            while time.time() - start < duration:
                if worker_stop.is_set():
                    return False
                time.sleep(0.1)
        finally:
            self._release_key("w")
        return True

    def _random_move_and_loot(self, max_wait: float) -> str:
        if keyboard is None and not hasattr(pyautogui, "keyDown"):
            log(f"{self.log_prefix} 无法发送按键。")
            return "stopped"

        log(f"{self.log_prefix} 顺序执行 W/A/S/D（每个 2 秒），并每 5 秒按一次 E（超时 {max_wait:.1f} 秒）。")
        start = time.time()
        last_e = start
        drop_ui_visible = False
        last_ui_log = 0.0
        min_drop_check_time = 10.0

        sequence = ["w", "a", "s", "d"]
        idx = 0
        active_key = None
        key_end_time = start

        try:
            while not worker_stop.is_set():
                now = time.time()

                if active_key is None or now >= key_end_time:
                    if active_key:
                        self._release_key(active_key)
                    active_key = sequence[idx]
                    idx = (idx + 1) % len(sequence)
                    self._press_key(active_key)
                    key_end_time = now + 2.0

                if now - last_e >= 5.0:
                    self._tap_key("e")
                    last_e = now

                if now - start >= min_drop_check_time:
                    if not drop_ui_visible:
                        if self._is_drop_ui_visible():
                            drop_ui_visible = True
                            log(f"{self.log_prefix} 检测到物品掉落界面，开始识别掉落物。")
                        else:
                            if now - last_ui_log > 3.0:
                                self._is_drop_ui_visible(log_detail=True)
                                last_ui_log = now
                    else:
                        if self._detect_and_pick_drop():
                            log(f"{self.log_prefix} 本波掉落已选择。")
                            return "ok"

                if now - start > max_wait:
                    log(f"{self.log_prefix} 超过 {max_wait:.1f} 秒未检测到掉落，判定卡死。")
                    return "timeout"

                time.sleep(0.1)

        finally:
            if active_key:
                self._release_key(active_key)

        return "stopped"

    def _press_key(self, key: str):
        try:
            if keyboard is not None:
                keyboard.press(key)
            else:
                pyautogui.keyDown(key)
        except Exception as e:
            log(f"{self.log_prefix} 按下 {key} 失败：{e}")

    def _release_key(self, key: str):
        try:
            if keyboard is not None:
                keyboard.release(key)
            else:
                pyautogui.keyUp(key)
        except Exception as e:
            log(f"{self.log_prefix} 松开 {key} 失败：{e}")

    def _tap_key(self, key: str):
        try:
            if keyboard is not None:
                keyboard.press_and_release(key)
            else:
                pyautogui.press(key)
        except Exception as e:
            log(f"{self.log_prefix} 发送 {key} 失败：{e}")

    def _is_drop_ui_visible(self, log_detail: bool = False, threshold: float = 0.7) -> bool:
        score, _, _ = match_template(BTN_CONFIRM_LETTER)
        if log_detail:
            log(f"{self.log_prefix} 掉落界面检查：确认选择 匹配度 {score:.3f}")
        return score >= threshold

    def _detect_and_pick_drop(self, threshold=0.8) -> bool:
        if click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 掉落确认：确认选择",
            threshold=0.7,
        ):
            time.sleep(1.0)
            return True
        return False

    def _anti_stuck_and_reset(self) -> bool:
        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            else:
                pyautogui.press("esc")
        except Exception as e:
            log(f"{self.log_prefix} 发送 ESC 失败：{e}")
        time.sleep(1.0)
        click_template("G.png", f"{self.log_prefix} 防卡死：点击 G.png", 0.6)
        time.sleep(1.0)
        click_template("Q.png", f"{self.log_prefix} 防卡死：点击 Q.png", 0.6)
        time.sleep(1.0)

        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 防卡死：再次进行按钮",
            25.0,
            0.8,
        ):
            log(f"{self.log_prefix} 防卡死：未能点击 再次进行.png。")
            return False
        return self._select_letter_sequence(f"{self.log_prefix} 防卡死", need_open_button=False)

    def _retreat_only(self):
        wait_and_click_template(BTN_RETREAT_START, f"{self.log_prefix} 撤退按钮", 20.0, 0.8)


class ModFragmentGUI(FragmentFarmGUI):
    def __init__(self, parent, cfg):
        self.cfg_key = "mod_guard_settings"
        self.letter_label = "mod密函"
        self.product_label = "mod成品"
        self.product_short_label = "mod成品"
        self.entity_label = "mod"
        self.letters_dir = MOD_DIR
        self.letters_dir_hint = "mod"
        self.preview_dir_hint = "mod"
        self.log_prefix = "[MOD]"
        super().__init__(parent, cfg)


class ModExpelGUI(ExpelFragmentGUI):
    def __init__(self, parent, cfg):
        self.cfg_key = "mod_expel_settings"
        self.letter_label = "mod密函"
        self.product_label = "mod成品"
        self.product_short_label = "mod成品"
        self.entity_label = "mod"
        self.letters_dir = MOD_DIR
        self.letters_dir_hint = "mod"
        self.preview_dir_hint = "mod"
        self.log_prefix = "[MOD-驱离]"
        super().__init__(parent, cfg)


# ======================================================================
#  main
# ======================================================================
def main():
    global app, uid_mask_manager
    cfg = load_config()

    root = tk.Tk()
    root.title("苏苏多功能自动化工具")
    start_ui_dispatch_loop(root)
    uid_mask_manager = UIDMaskManager(root)

    # 简单自适应分辨率 + DPI 缩放
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    try:
        base_h = 1080
        scale = max(1.0, min(1.5, sh / base_h))
        root.tk.call("tk", "scaling", scale)
    except Exception:
        pass

    win_w = min(1350, int(sw * 0.95))
    win_h = min(900, int(sh * 0.95))
    pos_x = (sw - win_w) // 2
    pos_y = (sh - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
    root.minsize(1000, 650)

    try:
        root.state("zoomed")
    except Exception:
        pass

    toolbar = ttk.Frame(root)
    toolbar.pack(fill="x", padx=10, pady=5)
    ttk.Button(toolbar, text="打开UID遮挡", command=lambda: uid_mask_manager.start()).pack(
        side="left", padx=4
    )
    ttk.Button(toolbar, text="关闭UID遮挡", command=lambda: uid_mask_manager.stop()).pack(
        side="left", padx=4
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    frame_firework = ttk.Frame(notebook)
    notebook.add(frame_firework, text="赛琪大烟花")
    app = MainGUI(frame_firework, cfg)

    frame_fragment = ttk.Frame(notebook)
    notebook.add(frame_fragment, text="人物碎片刷取")

    fragment_notebook = ttk.Notebook(frame_fragment)
    fragment_notebook.pack(fill="both", expand=True)

    frame_guard = ttk.Frame(fragment_notebook)
    fragment_notebook.add(frame_guard, text="探险无尽血清")
    guard_gui = FragmentFarmGUI(frame_guard, cfg)
    register_fragment_app(guard_gui)

    frame_expel = ttk.Frame(fragment_notebook)
    fragment_notebook.add(frame_expel, text="驱离")
    expel_gui = ExpelFragmentGUI(frame_expel, cfg)
    register_fragment_app(expel_gui)

    frame_mod = ttk.Frame(notebook)
    notebook.add(frame_mod, text="mod刷取")

    mod_notebook = ttk.Notebook(frame_mod)
    mod_notebook.pack(fill="both", expand=True)

    mod_guard_frame = ttk.Frame(mod_notebook)
    mod_notebook.add(mod_guard_frame, text="探险无尽血清")
    mod_guard_gui = ModFragmentGUI(mod_guard_frame, cfg)
    register_fragment_app(mod_guard_gui)

    mod_expel_frame = ttk.Frame(mod_notebook)
    mod_notebook.add(mod_expel_frame, text="驱离")
    mod_expel_gui = ModExpelGUI(mod_expel_frame, cfg)
    register_fragment_app(mod_expel_gui)

    fragment_gui_map = {
        frame_guard: guard_gui,
        frame_expel: expel_gui,
        mod_guard_frame: mod_guard_gui,
        mod_expel_frame: mod_expel_gui,
    }

    fragment_notebooks = [
        (frame_fragment, fragment_notebook),
        (frame_mod, mod_notebook),
    ]

    def update_active_fragment_gui(event=None):
        current_main = notebook.select()
        if not current_main:
            set_active_fragment_gui(None)
            return
        main_widget = notebook.nametowidget(current_main)
        for container, sub_nb in fragment_notebooks:
            if main_widget is container:
                current_sub = sub_nb.select()
                if not current_sub:
                    set_active_fragment_gui(None)
                    return
                frame = sub_nb.nametowidget(current_sub)
                gui = fragment_gui_map.get(frame)
                set_active_fragment_gui(gui)
                return
        set_active_fragment_gui(None)

    fragment_notebook.bind("<<NotebookTabChanged>>", update_active_fragment_gui)
    mod_notebook.bind("<<NotebookTabChanged>>", update_active_fragment_gui)
    notebook.bind("<<NotebookTabChanged>>", update_active_fragment_gui)
    update_active_fragment_gui()

    def on_close():
        if uid_mask_manager is not None:
            uid_mask_manager.stop(manual=False, silent=True)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    log("苏苏多功能自动化工具 已启动。")
    root.mainloop()


if __name__ == "__main__":
    main()
