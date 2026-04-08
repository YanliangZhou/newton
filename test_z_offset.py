import warp as wp
import numpy as np

# 模拟两个不同的target_pos
test_cases = [
    ("Z=0", wp.vec3(0.0, 80.0, 0.0)),
    ("Z=-300", wp.vec3(0.0, 80.0, -300.0))
]

# 模拟一个简单的顶点
center = [-617.5, 1347.3, 200.0]
vertices = [wp.vec3(-617.5, 1347.3, 200.0)]

for name, target_pos in test_cases:
    print(f"\n=== 测试 {name} ===")
    print(f"目标位置: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
    
    scale_factor = 0.01
    height_scale = 1.2
    
    # 计算缩放后的中心点
    scaled_center = [
        center[0] * scale_factor,
        center[1] * scale_factor * height_scale,
        center[2] * scale_factor
    ]
    
    print(f"缩放后中心: ({scaled_center[0]:.2f}, {scaled_center[1]:.2f}, {scaled_center[2]:.2f})")
    
    # 计算从缩放中心到目标位置的偏移
    final_offset = [
        target_pos[0] - scaled_center[0],
        target_pos[1] - scaled_center[1],
        target_pos[2] - scaled_center[2]
    ]
    
    print(f"最终偏移: ({final_offset[0]:.2f}, {final_offset[1]:.2f}, {final_offset[2]:.2f})")
    
    # 应用到顶点
    for v in vertices:
        scaled_v = wp.vec3(
            v[0] * scale_factor,
            v[1] * scale_factor * height_scale,
            v[2] * scale_factor
        )
        adjusted_v = wp.vec3(
            scaled_v[0] + final_offset[0],
            scaled_v[1] + final_offset[1],
            scaled_v[2] + final_offset[2]
        )
        print(f"最终顶点位置: ({adjusted_v[0]:.2f}, {adjusted_v[1]:.2f}, {adjusted_v[2]:.2f})")
