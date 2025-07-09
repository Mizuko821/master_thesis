# 日本語感情分析におけるドメイン適応研究

## 🎯 研究の背景・目的

### 解決したい課題
機械学習の実運用では、**学習時のデータ（ソースドメイン）**と**実際の運用時のデータ（ターゲットドメイン）**が異なるドメインである場合が多く、この**ドメインギャップ**により性能が大幅に低下する問題があります。

本研究では、日本語の感情分析タスクにおいて：
- **ソースドメイン**: 企業の決算報告書（フォーマルな文書）
- **ターゲットドメイン**: SNS（Twitter）のテキスト（インフォーマルな文書）

という設定で、この現実的な課題に対するアプローチを検証しました。

### ビジネス上の意義
- **実用性**: 既存の企業データで学習したモデルをSNS監視に活用
- **コスト効率**: 新規ドメインでのアノテーションコスト削減
- **汎用性**: ECレビュー、顧客サポート、市場分析など多分野への応用

## 🔬 技術的アプローチ

### 基本戦略
従来のシングルタスク学習に対して、以下2つの手法でドメイン適応性能の向上を目指しました：

1. **マルチタスク学習**: 感情分析と同時にカテゴリ分類を学習することで、より汎用的な特徴表現を獲得
2. **敵対的学習（GRL）**: Gradient Reversal Layerを用いてドメイン不変な特徴を学習

### 使用技術スタック
- **深層学習フレームワーク**: PyTorch
- **自然言語処理**: Transformers (Hugging Face), 日本語BERT
- **データ処理**: pandas, numpy
- **評価・可視化**: scikit-learn, matplotlib, seaborn, t-SNE

## 📊 データセット設計

### 1. 企業決算報告書データセット（学習用）
```
構造: [target, sentence, category, polarity]
例: ["わが国経済", "当連結会計年度におけるわが国経済は...", "NULL#general", "neutral"]
```

**カテゴリ体系（24クラス）**:
- NULL系: amount, cost, general, price, profit, sales
- company系: amount, cost, general, profit, sales  
- business系: amount, cost, general, price, profit, sales
- product系: amount, cost, general, price, profit, sales
- market系: general

**感情極性（3クラス）**: positive, neutral, negative

### 2. 日本語Twitterデータセット（評価用）
CSVファイル形式で、改行コード除去などの前処理を実装

### データ分割戦略
ドメイン適応を厳密に評価するため、**カテゴリ別**にデータを分割：
- **学習用**: 12カテゴリ（NULL, company, market系）
- **検証用**: 6カテゴリ（product系）
- **テスト用**: 6カテゴリ（business系）

## 🏗 モデルアーキテクチャ

### 1. シングルタスクモデル（ベースライン）
```
日本語BERT → Dropout → FC層 → ReLU → FC層 → ReLU → 感情分類（3クラス）
```

### 2. マルチタスクモデル
```
                 ┌→ FC → ReLU → FC → ReLU → 感情分類（3クラス）
日本語BERT → Pooler ┤
                 └→ FC → ReLU → FC → ReLU → カテゴリ分類（24クラス）
```

### 3. GRL+マルチタスクモデル（提案手法）
```
                 ┌→ FC → ReLU → FC → ReLU → 感情分類（3クラス）
日本語BERT → Pooler ┤
                 └→ GRL → FC → ReLU → FC → ReLU → カテゴリ分類（24クラス）
```

**GRL（Gradient Reversal Layer）の実装**:
```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_forward, scale):
        ctx.save_for_backward(scale)
        return input_forward
    
    @staticmethod
    def backward(ctx, grad_backward):
        scale, = ctx.saved_tensors
        return scale * -grad_backward, None  # 勾配を反転
```

## 🧪 実験設計

### 実験1: 基本性能評価（同一ドメイン内）
- **目的**: 各手法の基本性能を同一ドメイン内で比較
- **データ**: 企業決算報告書のみ
- **評価**: 精度、F1スコア、混同行列
- **可視化**: t-SNEによる特徴空間の可視化

### 実験2: ドメイン適応性能評価
- **目的**: ドメインギャップが性能に与える影響を定量化
- **設定**: 企業データで学習 → Twitterデータで評価
- **比較**: 3つの手法でドメイン適応性能を比較

### 実験3: クロスバリデーション
- **目的**: Twitterデータでの詳細な性能分析
- **手法**: 6分割クロスバリデーション
- **評価**: 統計的有意性の検証

## 💻 実装上の工夫

### カスタムデータセットクラス
```python
class TextDataset(Dataset):
    def __init__(self, target, sentence, category_labels, pol_labels):
        self.target = target.values
        self.sentence = sentence.values
        self.category_labels = category_labels.values
        self.pol_labels = pol_labels.values
```

### マルチタスク損失関数
```python
pol_loss = loss_fct(pred["pol_logits"], pol_labels)
category_loss = loss_fct(pred["category_logits"], category_labels)
total_loss = pol_loss + category_loss
```

### 評価フレームワーク
- 精度・F1スコア（macro平均）の自動計算
- 混同行列の可視化
- 誤分類例の詳細分析機能
- 学習過程のリアルタイム可視化

## 📁 ファイル構成・実行順序

```
実験1_シングルタスクモデル.ipynb       # 1. ベースライン実装
実験1_マルチタスクモデル.ipynb         # 2. マルチタスク学習実装  
実験1_GRL+マルチタスクモデル.ipynb     # 3. 敵対的学習実装

実験2_シングルタスク.ipynb            # 4. ドメイン適応評価（単一）
実験2_マルチタスク.ipynb              # 5. ドメイン適応評価（マルチ）
実験2_GRL+マルチタスク.ipynb          # 6. ドメイン適応評価（GRL）

実験3_シングルタスク.ipynb            # 7. クロスバリデーション（単一）
実験3_マルチタスク.ipynb              # 8. クロスバリデーション（マルチ）
実験3_GRL+マルチタスク.ipynb          # 9. クロスバリデーション（GRL）
```

## 🛠 技術的な学習ポイント

### 深層学習・NLP技術
- PyTorchでのカスタムモデル実装
- Hugging Face Transformersの活用
- 日本語BERTの利用とファインチューニング
- マルチタスク学習の設計・実装

### 研究・実験スキル
- 統計的に妥当な実験設計
- ドメイン適応の評価方法論
- 敵対的学習の理論と実装
- 再現可能な研究プロセス

### データ分析・エンジニアリング
- 大規模テキストデータの効率的処理
- カテゴリ別データ分割の戦略
- 評価指標の適切な選択・実装
- 可視化によるモデル解釈

---

このリポジトリは、実際のビジネス課題（ドメインギャップ）に対する技術的解決策を、体系的な実験を通じて検証した研究成果です。最新の深層学習技術を活用しつつ、実用性を重視した設計となっています。