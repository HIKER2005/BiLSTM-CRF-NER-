from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

doc = Document()

style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

style_p = style.paragraph_format
style_p.line_spacing = 1.5
style_p.space_after = Pt(6)

def add_heading_custom(text, level=2):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.font.name = '黑体'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    return h

def add_body(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.name = '宋体'
    run.font.size = Pt(12)
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    return p

def add_figure_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = '宋体'
    run.font.size = Pt(10.5)
    run.bold = True
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    return p

def add_figure_note(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run(text)
    run.font.name = '宋体'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(80, 80, 80)
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    return p

# ===== 标题页 =====
title = doc.add_heading('ERP 数据分析结果', level=1)
for run in title.runs:
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.name = '黑体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

add_body(
    '本文档呈现基于闪烁光视觉搜索任务的 EEG ERP 数据分析结果。实验包含三种刺激条件'
    '（A、B、C），被试在闪烁光背景下执行有目标视觉搜索任务。分析以视觉搜索阵列出现'
    '（标记 41）为时间零点，提取 −100 至 800 ms 的 ERP 分段，基线校正窗口为 −100 至 0 ms。'
    '为消除闪烁光诱发的稳态视觉诱发电位（SSVEP）对瞬态 ERP 成分的叠加干扰，对每种刺激条件'
    '分别计算正确反应试次与错误反应试次的差异波（correct − incorrect）。该减法基于 SSVEP '
    '由闪烁光物理频率驱动、与反应正误无关的假设，相减后 SSVEP 成分相消，保留的信号反映正确'
    '与错误反应之间的差异性认知 ERP 成分。所有平均后波形经 30 Hz 低通 Butterworth 零相位滤波'
    '（4 阶）以去除高频噪声。感兴趣电极为 PO7 和 PO8。共纳入 10 名被试。'
)

# ===== 2.1 图1 =====
add_heading_custom('2.1 SSVEP 消除后三种条件组平均 ERP 波形')

add_body(
    '图 1 展示了 SSVEP 消除后（correct − incorrect）三种刺激条件在各感兴趣电极上的组平均 '
    'ERP 波形。以视觉搜索阵列出现时刻为零点，分析窗口为 −100 至 800 ms。纵轴采用负值朝上'
    '的传统 ERP 绘图惯例。三种条件的波形形态总体相似，均可观察到典型的视觉加工成分（如 P1、'
    'N1）及晚期认知成分（如 N2、P3）。条件间的波形差异主要体现在 200 ms 之后的时间窗口内，'
    '提示不同刺激类型在注意分配和目标识别阶段可能存在差异性加工。'
)

add_figure_caption('图 1　三种刺激条件组平均 ERP 波形（SSVEP 已消除）')
add_figure_note(
    '注：各子图对应一个感兴趣电极（PO7、PO8）。红线 = A 条件（corr − incorr），蓝线 = B 条件'
    '（corr − incorr），黑线 = C 条件（corr − incorr）。灰色竖虚线标示刺激呈现时刻（0 ms），'
    '灰色水平线标示零电位。纵轴负值朝上。数据经 30 Hz 低通滤波。N = 10。'
)

# ===== 2.2 图2 =====
add_heading_custom('2.2 SSVEP 消除参考：正确 vs 错误 vs 差异波')

add_body(
    '图 2 展示了 SSVEP 消除的减法过程。对于各感兴趣电极，将三种刺激条件合并后分别计算'
    '正确反应试次和错误反应试次的组平均波形。正确反应波形（红色实线）和错误反应波形'
    '（蓝色虚线）均包含闪烁光驱动的 SSVEP 成分，二者叠加了相似的周期性振荡。差异波'
    '（黑色粗线 = correct − incorrect）中 SSVEP 成分被有效消除，波形更为平滑，呈现出清晰'
    '的瞬态 ERP 成分轮廓。这一结果验证了减法策略的有效性。'
)

add_figure_caption('图 2　SSVEP 消除参考图：正确反应、错误反应与差异波的对比')
add_figure_note(
    '注：各子图对应一个感兴趣电极。红色实线 = 正确反应试次组平均（含 SSVEP），蓝色虚线 = '
    '错误反应试次组平均（含 SSVEP），黑色粗线 = 差异波（correct − incorrect，SSVEP 已消除）。'
    '正确与错误波形均为三种刺激条件的合并平均。纵轴负值朝上。N = 10。'
)

# ===== 2.3 图3 =====
add_heading_custom('2.3 条件间差异波')

add_body(
    '图 3 展示了 SSVEP 消除后条件间的差异波。以 A 条件为基线，分别计算 B − A 和 C − A '
    '差异波，以直观揭示不同刺激条件之间的 ERP 振幅差异。差异波偏离零线的区段表明该时间'
    '段内两种条件间存在系统性的振幅差异。B − A 差异波和 C − A 差异波的极性、潜伏期及'
    '头皮分布差异可为探讨条件特异性认知加工机制提供参考。'
)

add_figure_caption('图 3　条件间差异波（B − A 和 C − A）')
add_figure_note(
    '注：各子图对应一个感兴趣电极。蓝线 = B − A 差异波，黑线 = C − A 差异波。差异波'
    '基于 SSVEP 消除后的数据计算。灰色竖虚线 = 0 ms，灰色水平线 = 零电位。纵轴负值朝上。N = 10。'
)

# ===== 3.2 图4 =====
add_heading_custom('2.4 各 ERP 成分地形图')

add_body(
    '图 4 展示了 SSVEP 消除后三种条件合并平均的 ERP 在四个典型成分潜伏期处的头皮电位'
    '分布（地形图）。P1（100 ms）和 N1（170 ms）成分呈现枕部为主的分布，反映早期视觉'
    '皮层对刺激的自动加工。N2（250 ms）成分在后部头皮区域表现突出，可能与视觉搜索中的'
    '注意选择过程相关。P3（400 ms）成分在顶-中央区域达到最大幅值，与目标识别及决策加工'
    '有关。'
)

add_figure_caption('图 4　各 ERP 成分地形图（SSVEP 已消除）')
add_figure_note(
    '注：从左至右分别为 P1（100 ms）、N1（170 ms）、N2（250 ms）和 P3（400 ms）时间点'
    '的头皮电位地形图。数据为三种刺激条件合并后的组平均。颜色编码代表电位幅值（μV），暖色'
    '= 正电位，冷色 = 负电位。N = 10。'
)

# ===== 3.3 图5 =====
add_heading_custom('2.5 各条件 N2 成分地形图')

add_body(
    '图 5 展示了三种刺激条件在 N2 时间窗口（250 ± 50 ms）内的平均 ERP 幅值地形分布。'
    '三种条件均在枕-顶区域呈现负向电位分布，与视觉搜索任务中注意引导和目标选择相关的 '
    'N2/N2pc 成分一致。条件间的地形分布差异可能反映了不同刺激类型对注意资源的差异性调用。'
)

add_figure_caption('图 5　各条件 N2 时间窗口地形图（SSVEP 已消除）')
add_figure_note(
    '注：从左至右为 A、B、C 三种条件在 250 ± 50 ms 时间窗口内的平均电位地形图。色标采用'
    '各图独立最大最小值映射。N = 10。'
)

# ===== 3.4 图6 =====
add_heading_custom('2.6 各条件 P3 成分地形图')

add_body(
    '图 6 展示了三种刺激条件在 P3 时间窗口（400 ± 100 ms）内的平均 ERP 幅值地形分布。'
    'P3 成分在三种条件下均呈现顶-中央区域的正向分布，与目标识别后的认知评估和决策过程'
    '有关。三种条件间 P3 地形分布的差异可能反映了不同刺激类型引发的认知加工负荷的不同。'
)

add_figure_caption('图 6　各条件 P3 时间窗口地形图（SSVEP 已消除）')
add_figure_note(
    '注：从左至右为 A、B、C 三种条件在 400 ± 100 ms 时间窗口内的平均电位地形图。色标采用'
    '各图独立最大最小值映射。N = 10。'
)

# ===== 3.5 图7 =====
add_heading_custom('2.7 ERP 地形图时空演变矩阵')

add_body(
    '图 7 以矩阵形式展示了三种刺激条件在 −100 至 800 ms 时间范围内的 ERP 头皮电位分布'
    '的时空演变过程。每行对应一种刺激条件（A、B、C），每列对应一个时间窗口（每 50 ms 一个，'
    '取中心时间 ± 25 ms 的平均值）。所有地形图采用全局统一色标，便于跨条件、跨时间点进行'
    '直接比较。从时间演变上可以观察到：刺激呈现后约 80–130 ms 枕区出现正向激活（P1），'
    '随后约 150–200 ms 出现负向偏转（N1），200–350 ms 后部区域出现 N2 成分，300 ms 后'
    '顶-中央区域逐渐出现 P3 正向成分。不同条件间的差异主要体现在 N2 和 P3 时间窗口内。'
)

add_figure_caption('图 7　三种条件 ERP 地形图时空演变矩阵（SSVEP 已消除）')
add_figure_note(
    '注：每行 = 一种刺激条件（A、B、C），每列 = 一个时间窗口（从 −100 至 800 ms，每 50 ms '
    '一个，各取 ± 25 ms 平均值）。所有地形图采用全局统一对称色标（μV），右侧 colorbar 标示'
    '幅值范围。暖色 = 正电位，冷色 = 负电位。数据为 SSVEP 消除后的组平均（N = 10）。'
)

# ===== 4.3 图8 =====
add_heading_custom('2.8 N2 平均振幅')

add_body(
    '图 8 展示了各感兴趣电极上三种刺激条件的 N2 平均振幅（250 ± 50 ms 时间窗口内的均值）。'
    '柱形图的高度表示组平均值，误差棒表示标准误（SE）。N2 振幅反映了视觉搜索过程中'
    '与注意选择相关的神经活动强度。各电极上三种条件的 N2 平均振幅及其相对大小关系可为'
    '后续统计检验提供参考。'
)

add_figure_caption('图 8　各电极 N2 平均振幅柱状图（SSVEP 已消除）')
add_figure_note(
    '注：各子图对应一个感兴趣电极。柱形高度 = 组平均 N2 平均振幅（250 ± 50 ms），误差棒 = '
    '标准误（SE）。红色 = A 条件，蓝色 = B 条件，黑色 = C 条件。N = 10。'
)

# ===== 4.3 图9 =====
add_heading_custom('2.9 P3 平均振幅')

add_body(
    '图 9 展示了各感兴趣电极上三种刺激条件的 P3 平均振幅（400 ± 100 ms 时间窗口内的均值）。'
    'P3 振幅通常被认为与认知资源分配、目标识别后的情境更新过程密切相关。条件间 P3 振幅的'
    '差异可能反映不同刺激类型在目标搜索过程中引发的认知加工深度的不同。'
)

add_figure_caption('图 9　各电极 P3 平均振幅柱状图（SSVEP 已消除）')
add_figure_note(
    '注：各子图对应一个感兴趣电极。柱形高度 = 组平均 P3 平均振幅（400 ± 100 ms），误差棒 = '
    '标准误（SE）。红色 = A 条件，蓝色 = B 条件，黑色 = C 条件。N = 10。'
)

# ===== 6.1 图10 =====
add_heading_custom('2.10 逐时间点重复测量方差分析')

add_body(
    '图 10 展示了各电极上逐时间点单因素重复测量方差分析的结果。上排为三种条件的组平均'
    '波形（SSVEP 已消除），下排为对应的 p 值曲线。灰色虚线标示未校正的显著性阈值'
    '（p = .05），红色实线标示经 FDR（False Discovery Rate）校正后的显著性阈值。p 值'
    '曲线低于 FDR 阈值的时间区段内，三种刺激条件间的 ERP 振幅存在经多重比较校正后的'
    '统计学显著差异。该分析以探索性方式揭示了条件效应在时间维度上的动态变化模式。'
)

add_figure_caption('图 10　逐时间点重复测量方差分析结果（SSVEP 已消除）')
add_figure_note(
    '注：上排各子图为三种刺激条件（A = 红，B = 蓝，C = 黑）在各电极上的组平均 ERP 波形。'
    '下排各子图为对应电极上逐时间点单因素重复测量方差分析的 p 值曲线。灰色虚线 = p = .05'
    '（未校正），红色实线 = FDR 校正阈值。p 值低于红线的时间段表示三种条件间差异达到 '
    'FDR 校正后的显著性水平。分析基于 SSVEP 消除后经 30 Hz 低通滤波的数据。N = 10。'
)

# ===== 8.1 图11 =====
add_heading_custom('2.11 分条件 SSVEP 消除过程')

add_body(
    '图 11 分别展示了三种刺激条件（A、B、C）在各感兴趣电极上的 SSVEP 消除过程。对于'
    '每种条件，红色实线为正确反应试次的组平均波形（包含 SSVEP 成分），蓝色虚线为错误'
    '反应试次的组平均波形（同样包含 SSVEP），黑色粗线为二者的差异波（correct − incorrect）。'
    '可以观察到，正确和错误反应波形中均包含明显的周期性振荡（SSVEP），而差异波中该振荡'
    '被有效消除，证实了该减法策略对各刺激条件均具有良好的 SSVEP 去除效果。同时，差异波'
    '中保留了清晰的 ERP 成分结构（N2、P3 等），表明减法未损害瞬态 ERP 信息。'
)

add_figure_caption('图 11　各条件 SSVEP 消除过程：正确反应、错误反应与差异波')
add_figure_note(
    '注：行 = 电极（PO7、PO8），列 = 刺激条件（A、B、C）。红色实线 = 正确反应组平均'
    '（含 SSVEP），蓝色虚线 = 错误反应组平均（含 SSVEP），黑色粗线 = 差异波'
    '（correct − incorrect，SSVEP 已消除）。纵轴负值朝上。N = 10。'
)

# ===== 8.2 图12 =====
add_heading_custom('2.12 SSVEP 消除前后波形对比')

add_body(
    '图 12 直接对比了 SSVEP 消除前（原始正确反应波形）和消除后（correct − incorrect 差异波）'
    '的波形形态。对于各电极，细实线为消除前的原始正确反应波形，粗虚线为消除后的差异波。'
    '可以清晰地观察到：消除前波形中叠加的高频 SSVEP 振荡在消除后被显著抑制，波形变得'
    '平滑，ERP 成分的辨识度明显提高。三种条件在消除前后的波形变化模式相似，表明 SSVEP '
    '消除过程未引入条件特异性的系统偏差。'
)

add_figure_caption('图 12　SSVEP 消除前后波形对比')
add_figure_note(
    '注：各子图对应一个感兴趣电极。细实线 = 消除前原始正确反应波形（含 SSVEP），粗虚线 = '
    '消除后差异波（correct − incorrect）。红色 = A 条件，蓝色 = B 条件，黑色 = C 条件。'
    '纵轴负值朝上。N = 10。'
)

# ===== 8.3 图13 =====
add_heading_custom('2.13 SSVEP 消除后各成分地形图')

add_body(
    '图 13 展示了 SSVEP 消除后（correct − incorrect）三种条件合并平均的 ERP 在 P1（100 ms）、'
    'N1（170 ms）、N2（250 ms）和 P3（400 ms）四个典型成分时间点的头皮电位地形分布。'
    '与图 4 相同，本图进一步确认了各 ERP 成分的空间分布特征：早期成分（P1、N1）集中于'
    '枕部，N2 分布于后部头皮，P3 在顶-中央区域最为突出。该地形图模式与视觉搜索任务中的'
    '经典 ERP 文献报道一致，支持了 SSVEP 消除后所得波形的生理学合理性。'
)

add_figure_caption('图 13　SSVEP 消除后各 ERP 成分地形图')
add_figure_note(
    '注：从左至右分别为 P1（100 ms）、N1（170 ms）、N2（250 ms）和 P3（400 ms）时间点'
    '的头皮电位地形图。数据为三种条件合并后的组平均差异波（correct − incorrect）。颜色编码'
    '代表电位幅值（μV），暖色 = 正电位，冷色 = 负电位。N = 10。'
)

# ===== 保存 =====
output_path = '/workspace/ERP_数据分析结果图注说明.docx'
doc.save(output_path)
print(f'Word 文档已保存至: {output_path}')
