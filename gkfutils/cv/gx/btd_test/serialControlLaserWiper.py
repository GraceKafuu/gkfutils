import yaml
import os
import sys
import serial


def get_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


cfg = get_config(config_path="config/config.yaml")
cfg_base = cfg["base"]
cfg_log = cfg["log"]
cfg_alg = cfg["alg"]

DATA_TTYS1 = cfg_base["DATA_TTYS1"]
DATA_TTYS2 = cfg_base["DATA_TTYS2"]
DATA_TTYS3 = cfg_base["DATA_TTYS3"]

wiper_move_gap = cfg_base["wiper_move_gap"]
wiper_move_gap_bytes = int(wiper_move_gap).to_bytes(2, byteorder="big")
wiper_move_gap_str = ' '.join(f'{b:02X}' for b in wiper_move_gap_bytes)
open_laser_cmd = bytes.fromhex(cfg_base["open_laser_cmd"].replace(' ', ''))
close_laser_cmd = bytes.fromhex(cfg_base["close_laser_cmd"].replace(' ', ''))
open_wiper_cmd = bytes.fromhex(cfg_base["open_wiper_cmd"].replace(' ', '') + ' ' + wiper_move_gap_str)
close_wiper_cmd = bytes.fromhex(cfg_base["close_wiper_cmd"].replace(' ', ''))


if __name__ == "__main__":
    serialPort = serial.Serial(DATA_TTYS1[0], DATA_TTYS1[1], timeout=DATA_TTYS1[2], parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)

    if sys.argv[1] == "open":
        serialPort.write(open_laser_cmd)
        serialPort.write(open_wiper_cmd)
        print("写入串口成功！开启激光：{}".format(open_laser_cmd))
        print("写入串口成功！开启雨刮器：{}".format(open_wiper_cmd))

    elif sys.argv[1] == "close":
        serialPort.write(close_laser_cmd)
        serialPort.write(close_wiper_cmd)
        print("写入串口成功！关闭激光：{}".format(open_laser_cmd))
        print("写入串口成功！关闭雨刮器：{}".format(open_wiper_cmd))

    else:
        print("请输入参数: open或者close!")
