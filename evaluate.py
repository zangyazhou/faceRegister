# coding=utf-8
"""
人脸识别系统 · 性能实证评估工具

用途：用真实人脸照片，量化验证系统的识别能力，论证匹配阈值（默认 0.55）的合理性。
这是毕设「实验」章节的数据来源——产出识别率、误识率(FAR)、拒识率(FRR)、
ROC 曲线、混淆矩阵与距离分布，可直接放入论文。

数据准备（目录结构，全部放照片，支持 jpg/jpeg/png）：
    data/evaluation/
        enrolled/           # 已注册学生，每人一个子文件夹，第一张照片作为注册模板，
            孟汉君/          #   其余照片作为「正样本」用于识别测试
                1.jpg
                2.jpg
                3.jpg
            学生B/
                1.jpg
                2.jpg
        impostor/           # 陌生人（负样本），每人一个子文件夹，全部用于误识测试
            陌生人A/
                1.jpg

运行：
    python evaluate.py                    # 使用默认阈值 0.55
    python evaluate.py --threshold 0.6    # 指定阈值

输出：
    evaluation_report/ 目录下：report.txt（文本报告）、距离分布图、FAR/FRR 曲线、
    ROC 曲线、混淆矩阵图、distances.csv（原始距离数据）
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗，直接写文件
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from api import face_encodings, load_image_file

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', 'evaluation')
REPORT_DIR = os.path.join(BASE, 'evaluation_report')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def list_images(folder):
    return [os.path.join(folder, n) for n in sorted(os.listdir(folder))
            if n.lower().endswith(IMAGE_EXTS)]


def extract_encoding(path):
    """提取图片中第一张人脸的 128 维编码；失败（无脸/读图失败）返回 None。"""
    try:
        img = load_image_file(path)
        encs = face_encodings(img)
        return np.asarray(encs[0], dtype=np.float64) if encs else None
    except Exception:
        return None


def load_dataset(data_dir):
    """
    加载评估数据集。
    :return: gallery(注册模板 [(label, enc)]), positives(正样本 [(label, enc)]),
             negatives(负样本 [(label='陌生人', enc)])
    """
    enrolled_dir = os.path.join(data_dir, 'enrolled')
    impostor_dir = os.path.join(data_dir, 'impostor')
    gallery, positives, negatives = [], [], []

    if not os.path.isdir(enrolled_dir):
        print('[错误] 未找到目录 %s，请先按本文件头部的说明准备照片' % enrolled_dir)
        return None

    for person in sorted(os.listdir(enrolled_dir)):
        folder = os.path.join(enrolled_dir, person)
        if not os.path.isdir(folder):
            continue
        images = list_images(folder)
        if not images:
            continue
        g = extract_encoding(images[0])
        if g is None:
            print('  [跳过] 注册照片未检出人脸：%s' % images[0])
            continue
        gallery.append((person, g))
        for img in images[1:]:
            e = extract_encoding(img)
            if e is not None:
                positives.append((person, e))
            else:
                print('  [跳过] 未检出人脸：%s' % img)

    if os.path.isdir(impostor_dir):
        for person in sorted(os.listdir(impostor_dir)):
            folder = os.path.join(impostor_dir, person)
            if not os.path.isdir(folder):
                continue
            for img in list_images(folder):
                e = extract_encoding(img)
                if e is not None:
                    negatives.append(('陌生人', e))
                else:
                    print('  [跳过] 未检出人脸：%s' % img)

    return gallery, positives, negatives


def compute_metrics(gallery, positives, negatives):
    """计算全部评估指标。"""
    g_labels = [l for l, _ in gallery]
    g_encs = np.array([e for _, e in gallery])

    # 1) 正样本对距离（同一人 probe 到其注册模板的距离）
    genuine_dists = []
    for label, enc in positives:
        if label in g_labels:
            gi = g_labels.index(label)
            genuine_dists.append(float(np.linalg.norm(enc - g_encs[gi])))

    # 2) 负样本对距离（陌生人 probe 到最近注册模板的距离）
    impostor_dists = []
    for label, enc in negatives:
        d = np.linalg.norm(g_encs - enc, axis=1)
        impostor_dists.append(float(np.min(d)))

    # 3) Rank-1 识别率（正样本在注册集找最近邻，看是否本人）
    correct = 0
    for label, enc in positives:
        pred = int(np.argmin(np.linalg.norm(g_encs - enc, axis=1)))
        if g_labels[pred] == label:
            correct += 1
    rank1 = correct / len(positives) if positives else 0.0

    return g_labels, genuine_dists, impostor_dists, rank1


def far_frr_at(genuine, impostor, t):
    """给定阈值 t，返回 (FAR, FRR)。FAR=负样本误接受比例，FRR=正样本误拒绝比例。"""
    far = sum(1 for d in impostor if d < t) / len(impostor) if impostor else 0.0
    frr = sum(1 for d in genuine if d >= t) / len(genuine) if genuine else 0.0
    return far, frr


def confusion_matrix(gallery, positives, negatives, tolerance):
    """多分类混淆矩阵：真实身份 × 预测身份（含「陌生人/拒绝」类）。"""
    g_labels = [l for l, _ in gallery]
    g_encs = np.array([e for _, e in gallery])
    labels = g_labels + ['陌生人/拒绝']

    # 真实身份 → 预测身份 计数
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    def predict(enc):
        if len(g_encs) == 0:
            return '陌生人/拒绝'
        d = np.linalg.norm(g_encs - enc, axis=1)
        best = int(np.argmin(d))
        return g_labels[best] if d[best] <= tolerance else '陌生人/拒绝'

    for label, enc in positives:
        cm[labels.index(label)][labels.index(predict(enc))] += 1
    for label, enc in negatives:
        cm[labels.index('陌生人/拒绝')][labels.index(predict(enc))] += 1
    return labels, cm


def plot_distance_distribution(genuine, impostor, tolerance, path):
    plt.figure(figsize=(8, 5))
    if genuine:
        plt.hist(genuine, bins=25, alpha=0.55, color='#10b981', density=True, label='正样本（同一人）')
    if impostor:
        plt.hist(impostor, bins=25, alpha=0.55, color='#ef4444', density=True, label='负样本（陌生人）')
    plt.axvline(tolerance, color='#4f46e5', linestyle='--', linewidth=2, label='阈值 %.2f' % tolerance)
    plt.xlabel('欧氏距离', fontsize=11)
    plt.ylabel('密度', fontsize=11)
    plt.title('人脸特征距离分布（正样本 vs 负样本）', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_far_frr(genuine, impostor, tolerance, path):
    thresholds = np.linspace(0.30, 0.90, 61)
    far_s, frr_s = [], []
    for t in thresholds:
        far, frr = far_frr_at(genuine, impostor, t)
        far_s.append(far)
        frr_s.append(frr)
    # 等错误率 EER
    diffs = np.abs(np.array(far_s) - np.array(frr_s))
    eer_i = int(np.argmin(diffs))
    eer = (far_s[eer_i] + frr_s[eer_i]) / 2

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far_s, color='#ef4444', label='FAR 误识率', linewidth=2)
    plt.plot(thresholds, frr_s, color='#f59e0b', label='FRR 拒识率', linewidth=2)
    plt.axvline(tolerance, color='#4f46e5', linestyle='--', linewidth=2, label='阈值 %.2f' % tolerance)
    plt.axhline(eer, color='#64748b', linestyle=':', linewidth=1.5, label='EER ≈ %.3f' % eer)
    plt.xlabel('阈值', fontsize=11)
    plt.ylabel('错误率', fontsize=11)
    plt.title('FAR / FRR 随阈值变化', fontsize=13)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return eer


def plot_roc(genuine, impostor, path):
    thresholds = np.linspace(0.30, 0.90, 61)
    tpr_s, fpr_s = [], []
    for t in thresholds:
        far, frr = far_frr_at(genuine, impostor, t)
        fpr_s.append(far)
        tpr_s.append(1 - frr)  # TPR = 1 - FRR
    # 按 FPR 排序画 ROC
    order = np.argsort(fpr_s)
    plt.figure(figsize=(6, 6))
    plt.plot([fpr_s[i] for i in order], [tpr_s[i] for i in order], color='#4f46e5', linewidth=2)
    plt.plot([0, 1], [0, 1], color='#cbd5e1', linestyle='--', label='随机')
    plt.xlabel('FPR（误识率）', fontsize=11)
    plt.ylabel('TPR（识别率）', fontsize=11)
    plt.title('ROC 曲线', fontsize=13)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(labels, cm, path):
    plt.figure(figsize=(max(6, len(labels) * 1.4), max(5, len(labels) * 1.2)))
    plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(label='样本数')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(range(len(labels)), labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center',
                     color='white' if cm[i][j] > cm.max() / 2 else '#1e293b', fontsize=11)
    plt.xlabel('预测身份', fontsize=11)
    plt.ylabel('真实身份', fontsize=11)
    plt.title('识别混淆矩阵（阈值 %.2f）' % 0.55, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='人脸识别系统性能评估')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='匹配阈值（默认 0.55，与系统 FACE_MATCH_TOLERANCE 一致）')
    parser.add_argument('--data-dir', default=DATA_DIR, help='评估数据目录')
    args = parser.parse_args()
    tolerance = args.threshold

    print('=' * 60)
    print('人脸识别系统 · 性能评估')
    print('数据目录：%s' % args.data_dir)
    print('=' * 60)

    loaded = load_dataset(args.data_dir)
    if loaded is None:
        return
    gallery, positives, negatives = loaded

    if not gallery:
        print('[错误] enrolled/ 中没有可用的人脸，无法评估')
        return
    if not positives and not negatives:
        print('[错误] 没有正/负样本照片，请按说明补充数据')
        return

    g_labels, genuine_dists, impostor_dists, rank1 = compute_metrics(
        gallery, positives, negatives)

    print('\n【数据集规模】')
    print('  注册学生：%d 人（%s）' % (len(gallery), '、'.join(g_labels)))
    print('  正样本（同一人照片）：%d 张' % len(positives))
    print('  负样本（陌生人照片）：%d 张' % len(negatives))

    print('\n【核心指标 @ 阈值 %.2f】' % tolerance)
    far, frr = far_frr_at(genuine_dists, impostor_dists, tolerance)
    print('  Rank-1 识别率：%.2f%%' % (rank1 * 100))
    print('  误识率 FAR：%.2f%%（%d/%d 张陌生人被误接受）' % (
        far * 100,
        int(round(far * len(impostor_dists))) if impostor_dists else 0,
        len(impostor_dists)))
    print('  拒识率 FRR：%.2f%%（%d/%d 张正样本被误拒绝）' % (
        frr * 100,
        int(round(frr * len(genuine_dists))) if genuine_dists else 0,
        len(genuine_dists)))

    if genuine_dists:
        print('\n【距离统计】')
        print('  正样本距离：min=%.3f  mean=%.3f  max=%.3f' % (
            min(genuine_dists), np.mean(genuine_dists), max(genuine_dists)))
    if impostor_dists:
        print('  负样本距离：min=%.3f  mean=%.3f  max=%.3f' % (
            min(impostor_dists), np.mean(impostor_dists), max(impostor_dists)))
        if genuine_dists:
            margin = min(impostor_dists) - max(genuine_dists)
            print('  正负样本间隔（负样本最小距离 - 正样本最大距离）= %.3f%s' % (
                margin, '（分离良好）' if margin > 0.1 else '（存在重叠，需调低阈值）'))

    # 生成报告目录与图表
    os.makedirs(REPORT_DIR, exist_ok=True)
    plot_distance_distribution(genuine_dists, impostor_dists, tolerance,
                               os.path.join(REPORT_DIR, 'distance_distribution.png'))
    eer = plot_far_frr(genuine_dists, impostor_dists, tolerance,
                       os.path.join(REPORT_DIR, 'far_frr.png'))
    plot_roc(genuine_dists, impostor_dists, os.path.join(REPORT_DIR, 'roc.png'))
    labels, cm = confusion_matrix(gallery, positives, negatives, tolerance)
    plot_confusion_matrix(labels, cm, os.path.join(REPORT_DIR, 'confusion_matrix.png'))

    # 原始距离数据 CSV
    with open(os.path.join(REPORT_DIR, 'distances.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['类型', '距离'])
        for d in genuine_dists:
            w.writerow(['正样本', '%.6f' % d])
        for d in impostor_dists:
            w.writerow(['负样本', '%.6f' % d])

    # 文本报告
    report = []
    report.append('人脸识别系统性能评估报告')
    report.append('=' * 40)
    report.append('数据集：注册 %d 人 / 正样本 %d 张 / 负样本 %d 张' % (
        len(gallery), len(positives), len(negatives)))
    report.append('评估阈值：%.2f' % tolerance)
    report.append('')
    report.append('Rank-1 识别率：%.2f%%' % (rank1 * 100))
    report.append('误识率 FAR：%.2f%%' % (far * 100))
    report.append('拒识率 FRR：%.2f%%' % (frr * 100))
    report.append('等错误率 EER：%.3f' % eer)
    if genuine_dists:
        report.append('正样本距离范围：%.3f ~ %.3f（均值 %.3f）' % (
            min(genuine_dists), max(genuine_dists), np.mean(genuine_dists)))
    if impostor_dists:
        report.append('负样本距离范围：%.3f ~ %.3f（均值 %.3f）' % (
            min(impostor_dists), max(impostor_dists), np.mean(impostor_dists)))
    with open(os.path.join(REPORT_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print('\n【输出】评估报告与图表已生成到 %s' % REPORT_DIR)
    print('  - report.txt（文本报告）')
    print('  - distance_distribution.png（距离分布）')
    print('  - far_frr.png（FAR/FRR 曲线，EER ≈ %.3f）' % eer)
    print('  - roc.png（ROC 曲线）')
    print('  - confusion_matrix.png（混淆矩阵）')
    print('  - distances.csv（原始数据）')
    print('=' * 60)


if __name__ == '__main__':
    main()
