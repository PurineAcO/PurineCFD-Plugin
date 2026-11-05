# PurineCFD-Plugin

## 圆柱绕流（无粘二维势流，偶极子解）

`src/plotter/potential_flow_cylinder.py` 以「均匀来流 + 偶极子」叠加给出圆柱绕流的解析势流解
（$\psi=U_\infty\sin\theta\,(r-a^2/r)$），并输出流线、物面 $C_p$ 分布与定量自检报告：

```bash
python src/plotter/potential_flow_cylinder.py
```

产物位于 `results/cylinder-potential/`：

| 文件 | 内容 |
| --- | --- |
| `streamlines.png` | 最终流线图（含速度云图、驻点、物面） |
| `surface_pressure.png` | 物面压力系数 $C_p=1-4\sin^2\theta$ |
| `doublet_limit.png` | 源−汇对 → 偶极子极限（二阶收敛） |
| `laplace_check.png` | 极坐标有限差分求解 $\nabla^2\varphi=0$ 的独立验证 |
| `cp_surface.csv` / `summary.txt` | 物面数据与验证报告 |

推导与交互式演示见 `src/plotter/cyc.ipynb`。
