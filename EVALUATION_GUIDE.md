# Model Evaluation Guide for CIC-DDoS2019

## Understanding Your Results

With **severe class imbalance** (0.21% BENIGN vs 99.79% ATTACK), here's how to interpret your metrics:

### Key Metrics to Focus On

#### 1. **F1-Score** (Most Important for Imbalanced Data)
- **Good**: > 0.80
- **Excellent**: > 0.90
- **Why**: Balances precision and recall, better than accuracy for imbalanced data

#### 2. **Recall** (Detection Rate)
- **BENIGN Recall**: How many BENIGN samples are correctly identified
  - **Good**: > 0.70 (70% of BENIGN traffic detected)
  - **Excellent**: > 0.90
- **ATTACK Recall**: How many ATTACK samples are correctly identified
  - **Good**: > 0.95 (should be high since it's the majority class)

#### 3. **Precision** (False Positive Rate)
- **BENIGN Precision**: When model says BENIGN, how often is it correct?
  - **Good**: > 0.80
- **ATTACK Precision**: When model says ATTACK, how often is it correct?
  - **Good**: > 0.95

#### 4. **Accuracy** (Can Be Misleading!)
- **Warning**: High accuracy (>99%) with imbalanced data often means the model just predicts the majority class
- **Good**: > 0.95, BUT only if F1-score and Recall are also high

### Confusion Matrix Interpretation

```
      [[TN, FP]    ← Predicted BENIGN
       [FN, TP]]   ← Predicted ATTACK

TN (True Negative): Correctly predicted BENIGN
FP (False Positive): Predicted ATTACK but was BENIGN (BAD - missing attacks!)
FN (False Negative): Predicted BENIGN but was ATTACK (BAD - false alarms!)
TP (True Positive): Correctly predicted ATTACK
```

### What "Good" Results Look Like

#### Scenario 1: Good Model (Balanced Performance)
```
Class distribution: BENIGN=1055 (0.21%), ATTACK=498945 (99.79%)
Val acc:    0.9850
Precision:  0.9500
Recall:     0.8500  ← Good recall for minority class
F1-score:   0.8967
Confusion Matrix:
  [[TN=   800, FP=   255]  ← Most BENIGN detected
   [FN= 74842, TP=424103]]  ← Most ATTACK detected
```
✅ **This is good!** Model detects both classes reasonably well.

#### Scenario 2: Bad Model (Just Predicting Majority)
```
Class distribution: BENIGN=1055 (0.21%), ATTACK=498945 (99.79%)
Val acc:    0.9979  ← Very high!
Precision:  0.9980
Recall:     0.0100  ← TERRIBLE! Only 1% of BENIGN detected
F1-score:   0.0198  ← Very low!
Confusion Matrix:
  [[TN=    10, FP= 1045]  ← Almost all BENIGN missed!
   [FN=     0, TP=498945]] ← All ATTACK predicted correctly
```
❌ **This is bad!** Model just predicts ATTACK for everything.

### Red Flags to Watch For

1. **High Accuracy (>99%) but Low F1-Score (<0.50)**
   - Model is likely just predicting the majority class

2. **Very Low Recall for BENIGN (<0.10)**
   - Model can't detect the minority class
   - Most BENIGN traffic will be misclassified as ATTACK

3. **Very High FP (False Positives for BENIGN)**
   - Too many BENIGN samples predicted as ATTACK
   - High false alarm rate

4. **Very High FN (False Negatives for ATTACK)**
   - Too many ATTACK samples predicted as BENIGN
   - Missing real attacks!

### What Your Results Should Show

After applying class weights, you should see:

✅ **Improvement in BENIGN Recall**: Should increase from near 0% to >50%
✅ **Balanced F1-Score**: Should be >0.70 (ideally >0.80)
✅ **Lower but More Honest Accuracy**: May drop from 99.9% to 85-95%, but this is more realistic
✅ **Better Confusion Matrix**: More TN (correct BENIGN predictions)

### Next Steps if Results Are Poor

1. **If BENIGN Recall is still low (<0.50)**:
   - Increase class weights further
   - Use resampling (upsample BENIGN, downsample ATTACK)
   - Try different model architectures

2. **If F1-Score is low (<0.70)**:
   - Adjust class weights
   - Train for more epochs
   - Try different learning rates

3. **If Precision is too low**:
   - Model is too aggressive (too many false positives)
   - Adjust decision threshold (currently 0.5)
   - Increase regularization

### Ideal Target Metrics

For a production IDS system:
- **F1-Score**: > 0.85
- **BENIGN Recall**: > 0.75 (don't miss too many benign flows)
- **ATTACK Recall**: > 0.98 (catch almost all attacks)
- **Overall Accuracy**: > 0.90

Remember: **F1-Score and Recall are more important than Accuracy for imbalanced data!**

