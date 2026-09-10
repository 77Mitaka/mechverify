# mechverify

平面机构自动验证器 —— 可验证描述语言（IR）+ 分层验证器 + 数据回路。

目标：证明“自动验证器可作为机构设计的判据（oracle）”。给定一个平面机构的
描述（IR）与一组设计需求，自动判定其是否满足，并给出评分、逐项诊断与运动轨迹。

## 一键复现

```bash
# Windows
powershell -ExecutionPolicy Bypass -File reproduce.ps1
# Linux / macOS / Git Bash
bash reproduce.sh
```

脚本依次：安装依赖 → 运行测试 → 端到端 demo → 数据回路 → 生成 Modelica 模型 →
（可选）生成论文插图。

### 环境依赖

* Python ≥ 3.9，依赖见 `requirements.txt`（numpy / scipy / shapely / pytest）。
* **可选**（第 2 层精算）：OpenModelica（`omc`）与 PlanarMechanics 库。未安装时，
  除 `simulate`/omc 相关测试外其余功能照常运行（相关测试自动跳过）。
* 论文编译：XeLaTeX + ctex（`paper/paper.tex`）。

## 结构

```
mechverify/
  ir.py           机构中间表示（拓扑 + 几何 + 驱动）
  structural.py   第 0 层：结构/组合检查（自由度、Grashof、闭合可行性）
  kinematics.py   第 1 层（解析）：四杆闭环闭式解、装配分支、奇异、传递角、向量化扫描
  multibody.py    第 1 层（通用）：基于约束的多体运动学（转动副 + 移动副、多环），
                  阻尼牛顿 + 数值雅可比 + 延续法选支 + 最小奇异值（奇异裕度）
  sketch.py       由“参考位形草图”构建通用机构 IR（原理图编辑器应输出的形态）
  modelica.py     第 2 层（精算）：IR -> 自包含 Modelica 模型、一致初值、omc 后端
  requirement.py  需求 schema 与评分（传递角、曲柄整转、奇异裕度、直线度、目标轨迹）
  verifier.py     编排：verify(mechanism, requirement) -> VerifyResult
examples/         示例机构与需求、六杆构建脚本、直线机构搜索脚本、.mo 生成脚本
demo.py           端到端演示
tests/            pytest 测试
```

## 求解路径

* 四杆（简单 IR）走 `kinematics.py` 的解析快筛，毫秒级。
* 含 `anchors` 的通用机构（滑块、六杆等）走 `multibody.py` 的约束求解器。

## 使用

```bash
pip install -r requirements.txt
python demo.py
pytest -q
python examples/search_straightline.py   # 以验证器为评分器搜索近似直线机构
python examples/build_sixbar.py          # 由参考位形构建多环六杆并验证
python examples/emit_modelica.py         # 生成各示例的 Modelica 模型
```

## 复用现成工具的工作流（推荐，替代自造前端）

用现成 Modelica 工具（OMEdit/Dymola/MapleSim）画图、仿真与可视化，本项目只做
"导入 → 验证 → 导出"，不自造画布：

```bash
python -m mechverify import   model.mo  -o ir.json          # 工具导出 -> IR
python -m mechverify verify   ir.json   --req req.json      # 名义验证
python -m mechverify verify   ir.json   --req req.json --robust --samples 200
python -m mechverify export   ir.json   -o out.mo           # IR -> Modelica（回工具里看）
python -m mechverify simulate ir.json                        # 调 omc 直接仿真
```

`verify` 的退出码：通过为 0，失败为 1，便于脚本/CI 使用。

## Modelica 精算层（已跑通并交叉验证）

`simulate()` 自动定位 OpenModelica（PATH 或常见安装目录，如 `D:\OpenModelica`），
编译并仿真；未安装时给出明确报错，代码生成与初值仍可独立使用。

代码生成有**两个后端**（`export/simulate --backend`）：

| 后端 | 模块 | 形态 | 用途 |
|---|---|---|---|
| `planarmechanics`（默认） | `planar_mechanics.py` | 用 `PlanarMechanics` 库的 Body/FixedTranslation/Fixed/Revolute/Prismatic + connect 组件 | 可回 OMEdit/Dymola 拖拽、动画、仿真 |
| `selfcontained` | `modelica.py` | 自己写全约束方程，零依赖 | 轻量、可移植、便于与解析层逐式对照 |

PlanarMechanics 版已对四杆/直线机构/曲柄滑块/六杆全部编译仿真通过；曲柄滑块
与解析解误差 **5e-12**。`examples/emit_planar_mechanics.py` 批量生成。

**验证结果**：曲柄滑块模型经 omc 仿真，其滑块轨迹与解析解
`x = a·cosθ + √(b² − a²·sin²θ)` 的误差 < 1e-11（见
`tests/test_modelica_sim.py`）。该对照还查出并修复了生成器 y 坐标旋转公式
的符号错误。

> 注：本机并行构建（多个 clang 同时启动）会触发 `0xC0000142` DLL 初始化失败，
> 已把 `share/omc/scripts/Compile.bat` 的并行 `-j` 改为串行。若换机器遇到同类
> 错误，同样处理即可。

## 鲁棒性分析（第 3 层）

`robustness.verify_robust()` 对机构几何（杆长、铰点/锚点）施加制造误差的
蒙特卡洛扰动，报告需求通过率与各行为指标的分位数：

```python
from mechverify import verify_robust
res = verify_robust(mech, req, samples=100, rel_tol=0.002, seed=0)
print(res.summary())
```

示例（直线机构）：名义通过，但在 ±0.2% 误差下仅 **83%** 通过
（直线度 p95=0.0063 > 阈值 0.005）——说明它并不鲁棒。**名义 pass ≠ 鲁棒 pass**。

`tests/test_adversarial.py` 另含一组退化机构（非 Grashof 曲柄、闭合不可行、
传递角不达标、直线度不可能），确保验证器把它们正确判负。

## 数据回路（生成 → 验证 → 沉淀）

`corpus.Corpus` 是 SQLite 语料库，持久化 `(需求, 机构, 评分)` 三元组，并支持
JSONL 导入导出，闭合"生成—验证—沉淀"回路：

```bash
python examples/build_corpus.py            # 随机生成 300 个四杆 → 验证 → 入库
python -m mechverify corpus stats examples/corpus.sqlite
python -m mechverify corpus export examples/corpus.sqlite -o corpus.jsonl
```

示例结果：300 个随机四杆中 176 通过、124 失败，最高分 1.0。这就是数据驱动
机构综合所需的监督信号来源。

## 复用现成工具：Modelica -> IR

`convert.modelica_to_mechanism()` 把 Modelica 模型读回 IR（支持闭环）。两条路径：
优先读取模型内嵌的源 IR（生成器会写入 `// mechverify-ir:` 注释），否则解析
扁平化平面模型（转动副）。见 `tests/test_convert.py`。

```python
from mechverify import Mechanism, verify
mech = Mechanism.from_json_file("examples/fourbar_crank_rocker.json")
res = verify(mech, {"constraints": [{"type": "crank_rotatability", "hard": True},
                                     {"type": "transmission_angle_min", "value": 40.0, "hard": True}]})
print(res.summary())
```

## 说明

* 支持四杆（解析）与一般低副平面机构（转动副/移动副、多环，如曲柄滑块、六杆）。
* 第 0/1 层为自写求解器；第 2 层为 Modelica（PlanarMechanics 或自包含后端）；
  第 3 层为鲁棒性蒙特卡洛。
* 高副（齿轮/凸轮）暂未支持。
* 论文见 `paper/paper.tex`（PDF：`paper/paper.pdf`）。
