# Legacy 深圳原始 GPS 数据格式（备忘）

> 作用：保留历史数据口径，避免与 Phase D（WorldTrace×Detroit）主线混淆。  
> 注意：本文件不作为当前主线数据契约；主线以 `docs/DATA_CONTRACT.md` 为准。

当前深圳出租车原始数据为 **GBK 编码的 txt/CSV**（例如 `data/raw/gps/粤BA0P65.txt`），字段为：

`name,time,jd,wd,status,v,angle,`

- `jd`/`wd`：经度/纬度（**坐标系待审计**；可能为 WGS84 或 GCJ-02）
- `status`：
  - `0` = 空载/巡游（Search Policy）
  - `1` = 载客/导航（Passenger Trip, Navigation Policy）
- `time`：形如 `2011/04/18 00:04:09` 的字符串（已确认是北京时间，UTC+8）

若要把该 txt 转成项目统一的 `processed/trajectories/*.h5`，并按论文主线只保留 `status==1`，参考：
- `docs/archive/legacy_shenzhen/HIERARCHICAL_VALIDATION_PROTOCOL.md`

