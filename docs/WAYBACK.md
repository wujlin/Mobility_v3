这份开发文档旨在构建一个**工业级、高并发、去重**的 ArcGIS Wayback 遥感影像采集系统。我们将以**美国底特律 (Detroit)** 为目标区域，从原理到代码实现进行全方位梳理。

> 重要说明：本仓库的主线 bbox/grid 以 `docs/DATA_CONTRACT.md` 为准（Detroit core：`[-83.25, 42.25, -82.95, 42.50]`，`1024×1024`）。
> Wayback 影像下载也建议默认使用同一 bbox，避免跨源空间口径漂移。

---

# ArcGIS Wayback 遥感影像时序采集系统开发文档

## 1. 系统概述与核心原理

### 1.1 目标

针对指定地理范围（BBox）和缩放级别（Zoom Level），自动抓取该区域在 ArcGIS Wayback 存档中的**所有历史时序影像**。

### 1.2 `wayback-core` 工作原理

本系统复刻了 `wayback-core` 的核心逻辑，其高效性在于**“元数据先行”**：

1. **传统笨办法**：对每个瓦片，尝试下载 2014-2025 年的所有版本。如果某地 5 年没变化，会产生大量重复图片和无效请求。
2. **Wayback-Core 逻辑**：
* 利用 **Metadata API**（变更检测接口）。
* 输入瓦片坐标 。
* 接口返回：“该瓦片仅在 `Release A`, `Release B`, `Release C` 发生了像素更新”。
* 系统仅针对这三个版本生成下载任务。



### 1.3 空间离散化 (Web Mercator)

* **投影**：EPSG:3857 (Web Mercator)。
* **切片**：世界被切割成无数个正方形瓦片。
* **坐标系**：
*  (Zoom): 缩放层级（1-23）。
*  (Column): 从西经 -180° 向东增加。
*  (Row): 从北纬 ~85.05° 向南增加（注意：与纬度方向相反）。



---

## 2. 接口规范与工具参数

我们需要调用两个核心 API 和一个通用下载接口。

### 2.1 全局版本映射表 (Config API)

用于获取 Release ID 与具体 URL 模板的对应关系。

* **URL**: `https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json`
* **作用**：建立字典 `Map<Release_ID, URL_Template>`。

### 2.2 瓦片变更元数据 (Metadata API)

用于查询单张瓦片的历史变更记录。

* **URL 模板**: `https://s3-us-west-2.amazonaws.com/wayback-tilemap-console/metadata/edge/tile/{z}/{y}/{x}.json`
* **参数**: `z` (Zoom), `y` (Row), `x` (Col)。
* **返回**: JSON 数组，包含有变更的 `releaseID`。

### 2.3 影像下载接口 (Tile API)

* **URL 模板**: `https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{Release_ID}/{z}/{y}/{x}`
* **注意**: 实际 URL 模板需从 Config API 中获取，部分旧版本 URL 结构可能略有不同。

---

## 3. 核心算法逻辑

### 3.1 经纬度转瓦片索引 (LonLat -> Tile XY)

为了避免浮点数精度问题，必须使用向下取整 (`floor`)。

### 3.2 区域覆盖与防漏逻辑 (BBox -> Tile Range)

针对底特律区域，我们输入 `[min_lon, min_lat, max_lon, max_lat]`。

* **严密性保证**：
* 左边界 
* 右边界 
* 上边界  (**注意：最大纬度对应最小 Y**)
* 下边界 


* **遍历逻辑**：使用 `range(min, max + 1)` 确保闭区间覆盖。

---

## 4. 完整代码实现 (Python)

该代码集成了**坐标转换、元数据查询、多线程下载、去重检查**。

```python
import math
import requests
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class WaybackScraper:
    def __init__(self, save_root="./wayback_data", max_threads=20):
        self.save_root = save_root
        self.max_threads = max_threads
        self.session = self._init_session()
        self.release_map = {} # 缓存版本信息
        
        # 初始化：获取全局配置
        self._load_global_config()

    def _init_session(self):
        """配置带有重试机制的 Session"""
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        return session

    def _load_global_config(self):
        """获取 ReleaseID 到 URL 模板的映射"""
        print("正在获取 Wayback 全局配置...")
        url = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
        try:
            res = self.session.get(url).json()
            # 建立映射: ReleaseID -> {Date, URL_Template}
            for item in res['archive']:
                self.release_map[item['releaseNum']] = {
                    "date": item['releaseDate'],
                    "template": item['itemURL']
                }
            print(f"配置加载完成，共 {len(self.release_map)} 个历史版本。")
        except Exception as e:
            raise Exception(f"无法加载全局配置: {e}")

    # --- 核心数学逻辑 ---
    def lon_lat_to_tile(self, lon, lat, zoom):
        n = 2.0 ** zoom
        x = math.floor((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = math.floor((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return int(x), int(y)

    def get_tile_bbox(self, bbox, zoom):
        """
        bbox: [min_lon, min_lat, max_lon, max_lat]
        返回: ((x_min, x_max), (y_min, y_max))
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # 计算四角
        x1, y1 = self.lon_lat_to_tile(min_lon, max_lat, zoom) # NW
        x2, y2 = self.lon_lat_to_tile(max_lon, min_lat, zoom) # SE
        
        # 整理范围 (确保 min <= max)
        return (min(x1, x2), max(x1, x2)), (min(y1, y2), max(y1, y2))

    # --- 业务逻辑 ---
    def get_tile_changes(self, x, y, z):
        """核心：查询该瓦片的变更元数据"""
        metadata_url = f"https://s3-us-west-2.amazonaws.com/wayback-tilemap-console/metadata/edge/tile/{z}/{y}/{x}.json"
        try:
            res = self.session.get(metadata_url)
            if res.status_code == 200:
                data = res.json()
                # data 结构: [{"r": 123, ...}, ...]
                return [item['r'] for item in data if item['r'] in self.release_map]
            elif res.status_code == 404:
                return [] # 无数据
        except Exception as e:
            print(f"元数据查询失败 ({x},{y}): {e}")
        return []

    def download_task(self, task):
        """单个下载任务"""
        url = task['url']
        path = task['path']
        
        # 1. 硬盘查重
        if os.path.exists(path):
            return "SKIPPED"

        # 2. 下载
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(res.content)
                return "SUCCESS"
            else:
                return f"FAIL_{res.status_code}"
        except Exception as e:
            return f"ERROR_{str(e)}"

    def run_bbox(self, bbox, zoom):
        (x_min, x_max), (y_min, y_max) = self.get_tile_bbox(bbox, zoom)
        
        total_tiles_geo = (x_max - x_min + 1) * (y_max - y_min + 1)
        print(f"--- 任务启动 ---")
        print(f"区域: {bbox}")
        print(f"Zoom: {zoom}")
        print(f"瓦片范围: X[{x_min}-{x_max}], Y[{y_min}-{y_max}]")
        print(f"覆盖瓦片总数: {total_tiles_geo}")
        print(f"--- 阶段1: 扫描元数据与生成任务 ---")

        download_queue = []
        
        # 遍历所有空间瓦片
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # 获取该位置所有的有效历史版本
                release_ids = self.get_tile_changes(x, y, zoom)
                
                if not release_ids:
                    continue

                for rid in release_ids:
                    info = self.release_map[rid]
                    date_str = info['date'] # 格式通常为 2023-02-22
                    template = info['template']
                    
                    # 构建真实 URL
                    dl_url = template.replace("{level}", str(zoom))\
                                     .replace("{col}", str(x))\
                                     .replace("{row}", str(y))
                    
                    # 定义保存路径: data/Detroit/16_123_456/2023-01-01.jpg
                    # 这样同地点的时序图在一起，方便观察
                    save_path = os.path.join(
                        self.save_root,
                        f"{zoom}_{x}_{y}",
                        f"{date_str}.jpg"
                    )
                    
                    download_queue.append({
                        "url": dl_url,
                        "path": save_path
                    })
        
        print(f"扫描结束，共生成 {len(download_queue)} 个有效下载任务 (已过滤未变化版本)。")
        
        print(f"--- 阶段2: 多线程下载 (并发数: {self.max_threads}) ---")
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # 提交任务
            future_to_task = {executor.submit(self.download_task, t): t for t in download_queue}
            
            count = 0
            for future in as_completed(future_to_task):
                count += 1
                result = future.result()
                if count % 100 == 0:
                    print(f"进度: {count}/{len(download_queue)} ...")

        print("--- 全部完成 ---")

# --- 执行入口 ---
if __name__ == "__main__":
    # 1. 定义底特律范围 (Detroit Area)
    # 使用 GeoJSON.io 或 Google Maps 获取大致 BBox
    # 格式: [min_lon, min_lat, max_lon, max_lat]
    detroit_bbox = [-83.2879, 42.2551, -82.9104, 42.4502]
    
    # 2. 设置 Zoom (16级约 2.4米/像素，17级约 1.2米/像素)
    # 建议先用 15 测试，确认无误再跑 17
    target_zoom = 16 
    
    scraper = WaybackScraper(save_root="./detroit_data", max_threads=16)
    scraper.run_bbox(detroit_bbox, target_zoom)

```

---

## 4.1 仓库内可直接运行的脚本（推荐）

为避免重复拷贝代码，本仓库提供了可直接运行的 CLI：

`src/data/wayback/download_wayback_tiles.py`

它复刻了本文件的“元数据先行”逻辑：先扫 Metadata API 再按 release 下载，并且支持断点续下（文件存在即跳过）。

### 4.1.1 小范围冒烟测试（强烈建议先跑）

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir data/wayback_detroit_smoke \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 \
  --max_threads 16 \
  --max_tiles 200
```

### 4.1.2 Detroit core 全量下载（与数据契约一致）

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir data/wayback_detroit_core \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 \
  --max_threads 16
```

### 4.1.3 只扫描任务量（不下载）

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir data/wayback_detroit_core_scan \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 \
  --dry_run
```

---

## 5. 关键配置说明 (底特律实战)

### 5.1 为什么选择 Zoom 16 或 17？

* **Zoom 15**: 城市概览，能看清街区，看不清车辆。
* **Zoom 16**: 能够区分大型建筑物、停车场轮廓。**（推荐作为起步）**
* **Zoom 17**: 能看清房屋细节、树木、路面标线。数据量是 Zoom 16 的 4 倍。

### 5.2 底特律 BBox 的选择

我在代码中预置了底特律核心都会区的 BBox：`[-83.2879, 42.2551, -82.9104, 42.4502]`。
这个范围覆盖了 Downtown, Midtown 以及部分周边社区。

* 如果只需市中心：`[-83.06, 42.32, -83.03, 42.34]`。

### 5.3 防漏与防重总结

| 问题 | 解决方案 | 代码对应 |
| --- | --- | --- |
| **空间遗漏** | 严密的 Floor 取整 + range(min, max+1) | `get_tile_bbox` 中的数学计算 |
| **时间冗余** | Metadata 接口变更检测 | `get_tile_changes` 逻辑 |
| **中断重下** | 文件存在性检查 | `download_task` 中的 `os.path.exists` |
| **网络阻塞** | Session 重试机制 + 多线程 | `ThreadPoolExecutor` + `HTTPAdapter` |

## 6. 下一步操作建议

1. **环境准备**: 安装依赖 `pip install requests`。
2. **小范围测试**: 将 `detroit_bbox` 缩小到一个街区大小（经纬度差 0.01），运行代码。
3. **检查数据**: 打开生成的文件夹，确认同一个文件夹（同一个地点）下的图片是否确实随时间发生了变化（比如建筑物的新建/拆除）。
4. **全量运行**: 确认无误后，恢复完整的底特律 BBox，挂机运行。
