# 三模型原生验收：待最终审查

|模型|GPU/s|H0/s|原生成机制/s|旧创新三/s|旧完整系统设备/s|新系统设备/s|规划主机/s|GPU加速|较旧完整缩短|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|4B|132.966|162.372|107.623|80.825|68.263|69.053|0.935|1.926x|-1.16%|
|8B|178.512|245.501|187.004|149.550|103.641|103.259|0.931|1.729x|0.37%|
|14B|260.755|348.756|285.635|226.354|152.404|149.972|3.051|1.739x|1.60%|

offline controlled same-prompt proxy alpha1, current evaluation samples excluded; NOT chronological RL history

historically used test1 window, not pristine held-out

native device seconds separate from host planning; old Mac planning proxy excluded from old device comparison

54 wafer dies vs48 A100 baseline; not equal power/area/cost; calibrated trace cost simulation, no numerical training/learning or physical wafer

远端等价检查、最终瓶颈审查和证据交付由主任务继续；用户已暂缓GitHub上传，当前未标记整体完成。
