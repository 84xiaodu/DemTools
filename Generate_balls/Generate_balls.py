import random
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time

# 空间网格索引类


def get_grid_index(x, y, z, grid_size):
    return (int(x // grid_size), int(y // grid_size), int(z // grid_size))


class SpatialGrid:
    def __init__(self, box_size, grid_size):
        self.grid_size = grid_size
        self.box_size = box_size
        self.grid = defaultdict(list)

    def add_ball(self, x, y, z, r):
        idx = get_grid_index(x, y, z, self.grid_size)
        self.grid[idx].append((x, y, z, r))

    def get_nearby_balls(self, x, y, z):
        idx = get_grid_index(x, y, z, self.grid_size)
        balls = []
        # 检查本格及周围26个格子
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_idx = (idx[0]+dx, idx[1]+dy, idx[2]+dz)
                    balls.extend(self.grid.get(neighbor_idx, []))
        return balls

# 使用空间网格索引的重叠检测


def is_overlap_grid(x, y, z, r, grid):
    for bx, by, bz, br in grid.get_nearby_balls(x, y, z):
        d = math.sqrt((x-bx)**2 + (y-by)**2 + (z-bz)**2)
        if d < (r + br):
            return True
    return False

# 计算当前孔隙率


def calc_porosity(balls, box_size):
    V_box = box_size ** 3
    V_balls = sum(4/3 * math.pi * b[3]**3 for b in balls)
    return 1 - V_balls / V_box

# 并行采样辅助函数


def min_dist_for_points(points, balls, box_size, r_min_threshold):
    max_r = 0
    max_point = None
    for x, y, z in points:
        min_dist = min(x, box_size-x, y, box_size-y, z, box_size-z)
        for bx, by, bz, br in balls:
            d = math.sqrt((x-bx)**2 + (y-by)**2 + (z-bz)**2) - br
            if d < min_dist:
                min_dist = d
        if min_dist > max_r and min_dist >= r_min_threshold:
            max_r = min_dist
            max_point = (x, y, z)
    return max_r, max_point


def estimate_next_radius_adaptive(balls, box_size, r_min_threshold, coarse_step=2.0, fine_step=0.2, fine_range=2.0, timing_log=None):
    start_time = time.time()
    # 1. 粗采样
    points = [(x, y, z) for x in frange(0, box_size, coarse_step)
              for y in frange(0, box_size, coarse_step)
              for z in frange(0, box_size, coarse_step)]
    n_cpu = min(cpu_count(), 8)
    chunk_size = len(points) // n_cpu + 1
    chunks = [points[i*chunk_size:(i+1)*chunk_size] for i in range(n_cpu)]
    with Pool(n_cpu) as pool:
        results = pool.starmap(min_dist_for_points, [(
            chunk, balls, box_size, r_min_threshold) for chunk in chunks])
    # 找到全局最大空隙点
    max_r, max_point = max(results, key=lambda x: x[0])
    coarse_time = time.time() - start_time
    if timing_log is not None:
        timing_log.append(f"粗采样耗时: {coarse_time:.2f} 秒, 粗采样最大半径: {max_r:.3f}")
    if max_point is None:
        return max_r
    # 2. 局部细采样
    start_fine = time.time()
    x0, y0, z0 = max_point
    fine_points = [(x, y, z)
                   for x in frange(max(x0-fine_range, 0), min(x0+fine_range, box_size), fine_step)
                   for y in frange(max(y0-fine_range, 0), min(y0+fine_range, box_size), fine_step)
                   for z in frange(max(z0-fine_range, 0), min(z0+fine_range, box_size), fine_step)]
    chunk_size = len(fine_points) // n_cpu + 1
    fine_chunks = [
        fine_points[i*chunk_size:(i+1)*chunk_size] for i in range(n_cpu)]
    with Pool(n_cpu) as pool:
        fine_results = pool.starmap(min_dist_for_points, [(
            chunk, balls, box_size, r_min_threshold) for chunk in fine_chunks])
    fine_max_r, fine_max_point = max(fine_results, key=lambda x: x[0])
    fine_time = time.time() - start_fine
    if timing_log is not None:
        timing_log.append(
            f"细采样耗时: {fine_time:.2f} 秒, 细采样最大半径: {fine_max_r:.3f}")
    return min(fine_max_r, r_max)


def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

# 浮动r_min的多尺度填充法，使用空间网格索引加速重叠检测


def adaptive_multiscale_generate_balls(box_size, r_min, r_max, phi_target=0.5, max_attempts=20000, grid_step=1.0, timing_log=None):
    balls = []
    r = r_max
    grid_size = 2 * r_min  # 网格边长设为最小球半径
    grid = SpatialGrid(box_size, grid_size)
    while r >= r_min:
        attempts = 0
        added = False
        t0 = time.time()
        while attempts < max_attempts:
            x = random.uniform(0, box_size)
            y = random.uniform(0, box_size)
            z = random.uniform(0, box_size)
            if not is_overlap_grid(x, y, z, r, grid):
                balls.append([x, y, z, r])
                grid.add_ball(x, y, z, r)
                added = True
            attempts += 1
        fill_time = time.time() - t0
        if timing_log is not None:
            timing_log.append(
                f"半径 {r:.3f} 层级填充耗时: {fill_time:.2f} 秒, 新增球数: {attempts}")
        phi = calc_porosity(balls, box_size)
        if phi <= phi_target:
            break
        next_r = estimate_next_radius_adaptive(
            balls, box_size, r_min, coarse_step=2.0, fine_step=0.2, fine_range=2.0, timing_log=timing_log)
        if next_r < r_min:
            break
        r = next_r
        grid_size = max(r, r_min)
        grid = SpatialGrid(box_size, grid_size)
        for bx, by, bz, br in balls:
            grid.add_ball(bx, by, bz, br)
    phi = calc_porosity(balls, box_size)
    return balls, phi


if __name__ == "__main__":
    # 示例参数
    box_size = 10
    r_min = 0.2
    r_max = 0.5
    phi_target = 0.4  # 目标孔隙率
    t_start = time.time()
    timing_log = []
    balls, phi = adaptive_multiscale_generate_balls(
        box_size, r_min, r_max, phi_target, timing_log=timing_log)
    total_time = time.time() - t_start
    # 输出球体信息到文件
    with open('balls_output.txt', 'w', encoding='utf-8') as f:
        for ball in balls:
            f.write(
                f"{ball[0]:.6f} {ball[1]:.6f} {ball[2]:.6f} {ball[3]:.6f}\n")
    # 输出计时信息到文件
    with open('timing_output.txt', 'w', encoding='utf-8') as f:
        for line in timing_log:
            f.write(line + '\n')
        f.write(f"总耗时: {total_time:.2f} 秒\n")
    print(f"最终孔隙率: {phi:.4f}")
