# 毕业设计：基于深度学习的高校实验室人员全生命周期可信通行平台

LabGuard 将高速多人脸识别从单一“签到功能”扩展为高校实验室人员信息登记、组织归属、时段授权、门禁决策、考勤联动、风险审计和离组销毁的一体化平台。系统基于 Flask 3、OpenCV YuNet/SFace、MiniFASNetV2 与 dlib 关键点，采用质量驱动关键帧、轨迹级特征融合和风险自适应活体策略；历史 dlib 特征可无缝回退。

## 功能

- 管理员登录（密码哈希存储，历史明文密码自动升级）
- 人员全生命周期：在组、暂停、离组三态管理，支持成员类型、实验室归属、职责、入组/预计离组时间与备用认证方式
- 独立生物特征模板：人脸模板与个人档案分离存储并记录授权时间，不录脸也可先建档
- 实验室与门禁点：按实验室建立入口、设备区等门禁点，支持精确到门禁和有效时段的显式授权
- 可信通行决策：身份、人员状态、实验室成员关系、门禁授权、有效期和活体结果共同决定放行/拒绝
- 离组自动销毁：离组时物理删除个人档案、人脸模板、成员关系和授权，保留匿名审计事件与销毁回执
- 多人 Web 通行终端：同框多人独立跟踪和决策，支持门禁点参数，例如 `/web-sign-in?gate=DEMO-A305-DOOR`
- 隐私治理中心：集中展示模板持有情况、授权覆盖率、即将到期成员和可验证销毁回执
- 通行审计：记录允许/拒绝、拒绝原因、门禁点、识别轨迹和端到端时延
- 课程与班级、独立考勤场次、正常/迟到/缺勤业务判定
- 识别事件与风险事件审计：质量、相似度、候选间隔、活体概率、风险分和端到端时延
- LabGuard 数据驾驶舱：生命周期、实验室、权限覆盖、销毁回执、7 日趋势、决策构成、热力图、时延和模型健康度
- 算法实验中心：dlib/YuNet+SFace 性能对比、SFace 阈值曲线、FAR/FRR、证据完成度与样本限制
- 场次级明细与 CSV 导出：正常、迟到、缺勤及关联识别质量/相似度/活体/风险/时延
- 隔离答辩演示数据：清晰标记 `is_demo`，不生成或伪造任何生物特征，可一键切换回真实数据
- 本地 Apache ECharts + Bootstrap 5，断网答辩仍可完整展示

## 技术栈

- 后端：Flask 3 / Flask-SQLAlchemy / Flask-Migrate / Flask-Login / Flask-WTF
- 高速识别：YuNet 检测 + SFace 128维特征 + 余弦相似度/候选间隔开放集判定
- 活体检测：MiniFASNetV2 静默反欺骗；不确定时随机眨眼或转头一次
- 兼容链路：dlib HOG/ResNet 历史特征与68点动作检测
- 图像处理：OpenCV / Pillow / numpy
- 数据库：SQLite（可通过环境变量切换）
- 数据可视化：Apache ECharts 6.1（本地静态资源，不依赖公网）

## 环境要求

- **Python 3.10 ~ 3.12**：`numpy 1.26` 与 `dlib-bin 19.24` 只提供到 3.12 的预编译 wheel，Python 3.13+ 装不上。
- 国内网络建议先配置 pip 镜像：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`

## 环境准备

> 项目已在 Python 3.11 环境验证，依赖版本记录在 `requirements.txt`。
> 重新创建或换机时，按下面步骤操作即可。

1. 若项目目录存在残留的 `venv/`（本仓库曾附带一个 Linux 环境的 venv，在 Windows 上不可用），先删除再重建：

```bash
# Windows PowerShell
Remove-Item -Recurse -Force venv
```

2. 创建虚拟环境：

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖（`requirements.txt` 已选用跨平台的 `dlib-bin` 预编译 wheel，Windows/macOS/Linux 一次安装即可）：

```bash
pip install -r requirements.txt
```

4. 下载并校验 YuNet、SFace、MiniFASNetV2 模型：

```bash
python scripts/download_opencv_face_models.py
```

5. 复制 `.env.example` 为 `.env`，生成强随机密钥并设置初始管理员密码。

## 运行

```bash
# 1. 更新数据库（执行增量迁移）
flask --app app db upgrade

# 2. 初始化角色与初始管理员账号
flask --app app init

# 3. 启动服务
flask --app app run

# 或在 Windows 双击 run.bat（自动优先使用项目内 venv/.venv）
```

浏览器访问：
- 管理端：http://127.0.0.1:5000（需登录）
- **实验室可信通行终端（公开）**：http://127.0.0.1:5000/web-sign-in?gate=DEMO-A305-DOOR
- 实验室与门禁：http://127.0.0.1:5000/laboratories
- 通行审计：http://127.0.0.1:5000/access-events
- 隐私治理：http://127.0.0.1:5000/privacy-center
- 算法实验中心：http://127.0.0.1:5000/experiments

首次答辩演示可在「数据驾驶舱」点击「生成答辩演示数据」。系统会在隔离空间创建 28 名无生物特征的匿名成员、1 个实验室、2 个门禁点、时段授权、14 个历史考勤场次，以及带明确演示标识的通行/风险/时延数据；演示空间与真实数据空间可在左侧栏随时切换。

## 人员生命周期与可信通行闭环

```text
人员 User（在组 / 暂停 / 离组）
  ├─ 生物特征模板 FaceTemplate（独立存储 + 授权时间）
  ├─ 实验室成员关系 LabMembership（组织归属 + 有效期）
  ├─ 门禁授权 AccessGrant（门禁点 + 日期 + 每日时段）
  └─ 可信通行决策
       ├─ 识别事件 RecognitionEvent（身份/活体/质量/时延）
       ├─ 通行事件 AccessEvent（允许/拒绝 + 原因 + 门禁）
       └─ 风险事件 RiskEvent（照片攻击/身份冲突/审计证据）

离组 Offboarding
  └─ 原子化撤权与物理删除
       ├─ 删除人脸模板、档案、成员关系和授权
       └─ 保留匿名事件与 ErasureTask 销毁回执
```

识别事件只在一条人脸轨迹形成最终结论时写入，不逐帧记录，避免数据库日志膨胀。旧版 `SignIn` 每日签到表仍并行保留，用于兼容原有考勤查询与统计。

> ⚠️ 浏览器摄像头（`getUserMedia`）只在**安全上下文**下可用：`http://127.0.0.1` / `localhost` 或 HTTPS。
> 局域网 IP（如 `http://192.168.x.x:5000`）会被浏览器拦截摄像头，需配置 HTTPS 后才能用 Web 签到。

## 初始管理员账户

管理员账号固定为 `Admin`，初始密码必须在本机 `.env` 的
`ADMIN_PASSWORD` 中设置。仓库不包含真实数据库、密码、人脸图片或人脸特征；
首次登录后密码以哈希形式保存在本地数据库中。

## 隐私与数据说明

- `data.sqlite`、`.env`、上传照片、摄像头采集内容和人脸评估原图均被 Git 忽略。
- GitHub 中只保留匿名化指标，不包含人脸图片、人脸特征向量、真实姓名或学号。
- 换机后请自行准备有授权的数据集；不要将学生照片或生产数据库提交到仓库。

## 摄像头签到演示

```bash
# 有显示器：弹出识别窗口，识别到学生自动写入签到记录，按 q 退出
python test.py

# 无显示器环境（如服务器）：只识别与落库，不弹窗
python test.py --no-display
```

同一学生同日只记一条签到记录（数据库唯一约束兜底）。

## Web 端实验室可信通行（浏览器自助终端）

浏览器只负责获取摄像头画面，以二进制 JPEG 上传最新帧。检测、质量评价、
特征融合、活体、身份匹配、权限判定和审计落库均在后端完成；人脸编码不会下发到浏览器。

```
浏览器(getUserMedia，二进制 JPEG，最新帧优先)
   └─ POST /api/sign-in-image + CSRF 头
        └─ YuNet 多人检测 + IoU/中心距离轨迹关联
             ├─ 清晰度/光照/尺寸/姿态/置信度 → 质量关键帧
             ├─ SFace → 轨迹级质量加权特征融合 + 双阈值开放集判定
             ├─ MiniFASNetV2 → 静默活体风险评分
             ├─ 中风险 → 随机眨眼或转头一个动作（dlib 68点）
             └─ 活体通过且身份达到票数 → 人员状态/成员关系/门禁授权/有效期决策 → 独立落库
```

`static/vendor/face-api/` 是旧版浏览器识别方案的保留资源，当前 Web 签到页不再加载。

### 识别准确性设计（核心诉求：对的人能签、其他人绝不误签）

| 机制 | 说明 |
|---|---|
| **开放集双阈值** | SFace 同时要求最佳余弦相似度达到阈值、第一和第二候选之间保留足够间隔；低质量帧只会收紧阈值 |
| **质量关键帧** | 综合清晰度、曝光、尺寸、姿态和检测置信度，低质量帧只跟踪、不参与身份融合 |
| **轨迹特征融合** | 每条人脸轨迹保存紧凑的质量加权融合特征，降低单帧模糊、抖动导致的误判 |
| **多人独立轨迹** | 同框中的每张脸按位置跨帧关联，分别维护身份、票数与活体状态；陌生人不阻塞已注册人员 |
| **后端多帧确认** | 同一条人脸轨迹连续命中同一身份 3 票（`FACE_MATCH_VOTES`）才允许落库 |
| **风险自适应活体** | 静默活体、身份置信度、质量和时序稳定性共同计算风险；低风险无感通过，中风险随机一个动作，高风险拒绝 |
| **重复身份拒绝** | 同一身份在一帧中出现两次（如真人与其照片同框）时，两条轨迹都不允许签到 |
| **未注册明确拒绝** | 最近距离超过阈值即视为未注册，绝不落库 |
| **同日去重** | `record_sign_in` 唯一约束兜底，重复签到返回 `duplicate` |
| **最小权限策略** | 门禁模式默认拒绝，只有在组人员、有效成员关系与显式授权同时成立才允许通行 |
| **可验证销毁** | 离组删除身份数据与生物特征，匿名销毁回执记录删除范围和完成时间 |

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `FACE_MATCH_TOLERANCE` | `0.55` | 欧氏距离匹配阈值；越大越宽松，应根据本项目评估集的正负距离分布调整 |
| `FACE_MATCH_VOTES` | `3` | 落库前需连续命中的帧数（多帧确认） |
| `FACE_MATCH_WINDOW` | `6.0` | 多帧确认窗口（秒），超时票数作废 |
| `FACE_TRACK_TTL` | `2.0` | 漏检或离开后，人脸轨迹保留的秒数 |
| `FACE_ENGINE` | `auto` | 有 SFace 特征时自动使用高速流水线，否则回退 dlib |
| `SFACE_COSINE_THRESHOLD` | `0.40` | SFace 最低余弦相似度 |
| `SFACE_MINIMUM_MARGIN` | `0.08` | 最佳与次佳候选的最小间隔 |
| `FACE_MIN_QUALITY` | `0.38` | 进入身份特征提取的最低质量分 |
| `LIVENESS_REQUIRED` | `true` | 是否强制服务端活体检测 |
| `LIVENESS_POLICY` | `passive_adaptive` | 静默优先、随机单动作兜底 |
| `LIVENESS_WINDOW` | `8.0` | 活体决策/动作窗口（秒） |
| `LIVENESS_PASSIVE_MIN_FRAMES` | `2` | 静默活体最少高质量帧数 |
| `LIVENESS_PASSIVE_ACCEPT` | `0.80` | 无感放行阈值 |
| `LIVENESS_PASSIVE_REJECT` | `0.15` | 高风险拒绝阈值 |
| `LIVENESS_EAR_CLOSED` | `0.20` | EAR 闭眼阈值 |
| `LIVENESS_EAR_OPEN` | `0.24` | EAR 睁眼阈值 |
| `LIVENESS_HEAD_DELTA` | `0.12` | 判定转头所需的归一化鼻尖水平位移变化量 |

### 参数标定（推荐首次部署做）

运行 `python evaluate.py` 查看正/负样本距离分布，再调整 `FACE_MATCH_TOLERANCE`。
真人长期无法通过动作提示时，可根据实际摄像头帧中的 EAR/头部姿态适度调整活体阈值。

> 局限：普通 RGB 静默活体存在摄像头/环境域偏移，不能保证阻止所有高质量预录视频、
> 换脸或三维面具。当前系统因此保留随机动作兜底；高安全生产场景仍应增加红外或深度摄像头。

本机的可重复性能基线见 [`docs/performance_baseline.md`](docs/performance_baseline.md)。

### SFace 准确性基线

```bash
python scripts/evaluate_sface.py
```

结果写入 `evaluation_report/sface_metrics.json`，算法实验中心直接读取该文件。
当前数据仅有 1 个注册身份、2 个本人正样本和 11 个陌生人负样本，因此页面会把
结果标记为“功能性小样本基线”，不能把 100% Rank-1 外推为真实班级准确率。
后续多人、活体攻击与消融实验的执行口径见
[`docs/experiment_protocol.md`](docs/experiment_protocol.md)。

## 测试

```bash
python -m pytest -q
```

## 数据库迁移

```bash
# 升级到最新版本
flask --app app db upgrade

# 回退一步
flask --app app db downgrade
```

## 项目结构

```
app.py                     # Flask 应用：模型 / 表单 / 路由 / 权限
opencv_face_engine.py      # YuNet/SFace、质量评价、开放集余弦匹配
anti_spoof.py              # OpenCV DNN MiniFASNetV2 静默反欺骗
api.py                     # dlib 兼容链路与68点动作关键点
web_signin.py              # 多人轨迹/特征融合/风险自适应活体/落库
attendance_platform.py     # 课程考勤、最终识别事件、风险审计、驾驶舱聚合与演示数据
lifecycle_service.py        # 生命周期、成员关系、门禁授权、通行策略与离组销毁
experiment_service.py      # 读取性能/SFace实验产物并统一指标口径
benchmarks/                # 可重复运行的 dlib 与 OpenCV 性能对照
functions.py               # 人脸编码提取与安全解析
attendance.py              # 签到写入（防重复）
test.py                    # 摄像头实时识别签到
faceRecognitonModels/      # dlib 预训练模型
font/fontToImg.py          # OpenCV 中文标注
static/vendor/echarts/     # 本地 ECharts 驾驶舱图表（断网可用）
static/vendor/face-api/    # 旧版浏览器识别资源（当前签到页不加载）
templates/                 # Bootstrap 5 页面（含 web_sign_in.html 签到页）
migrations/                # Alembic 数据库迁移
tests/                     # pytest 测试（含多人/活体/权限策略/离组销毁测试）
```
