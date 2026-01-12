# Evaluation Metrics Documentation

## Overview

The Autonomous Defect Triage Agent uses a comprehensive set of evaluation metrics to assess the quality and performance of defect classification and root cause analysis. This document provides detailed explanations of each metric, their calculation methods, interpretation guidelines, and best practices.

## Table of Contents

1. [Core Classification Metrics](#core-classification-metrics)
2. [Confidence Metrics](#confidence-metrics)
3. [Performance Metrics](#performance-metrics)
4. [Category-Specific Metrics](#category-specific-metrics)
5. [Interpretation Guidelines](#interpretation-guidelines)
6. [Best Practices](#best-practices)

---

## Core Classification Metrics

### 1. Accuracy

**Definition**: The proportion of correct predictions among all predictions made.

**Formula**:
```
Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
```

**Range**: 0.0 to 1.0 (0% to 100%)

**Calculation in Code**:
```python
accuracy = correct_predictions / total_cases
```

**Interpretation**:
- **1.0 (100%)**: Perfect classification - all defects correctly categorized
- **0.9 (90%)**: Excellent - 9 out of 10 defects correctly categorized
- **0.7-0.8 (70-80%)**: Good - acceptable for most production systems
- **0.5-0.7 (50-70%)**: Fair - needs improvement
- **< 0.5 (< 50%)**: Poor - worse than random guessing

**When to Use**:
- Overall system performance assessment
- Balanced datasets where all categories are equally important
- Quick sanity check of model quality

**Limitations**:
- Can be misleading with imbalanced datasets
- Doesn't distinguish between types of errors (false positives vs false negatives)
- Treats all categories as equally important

**Example**:
```
Test cases: 100
Correct predictions: 85
Accuracy = 85/100 = 0.85 (85%)
```

---

### 2. Precision

**Definition**: Of all defects predicted as a certain category, what proportion were actually that category?

**Formula**:
```
Precision = True Positives / (True Positives + False Positives)
```

**Range**: 0.0 to 1.0 (0% to 100%)

**Calculation Method**:
The system uses **weighted precision** which accounts for class imbalance:
```python
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
```

**Interpretation**:
- **High Precision (> 0.9)**: When the agent predicts a category, it's usually correct
- **Medium Precision (0.7-0.9)**: Reasonably reliable predictions
- **Low Precision (< 0.7)**: Many false alarms - predicted category is often wrong

**When to Use**:
- When false positives are costly (e.g., alerting wrong teams)
- When you need high confidence in predictions
- When resources for investigating false alarms are limited

**Real-World Impact**:
- **High Precision**: Developers trust the system's categorization
- **Low Precision**: Developers ignore alerts due to frequent false positives

**Example**:
```
Database category predictions: 20
Actually database defects: 17
False alarms (not database): 3
Precision = 17/(17+3) = 0.85 (85%)
```

---

### 3. Recall (Sensitivity)

**Definition**: Of all defects that actually belong to a category, what proportion did we correctly identify?

**Formula**:
```
Recall = True Positives / (True Positives + False Negatives)
```

**Range**: 0.0 to 1.0 (0% to 100%)

**Calculation Method**:
The system uses **weighted recall**:
```python
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
```

**Interpretation**:
- **High Recall (> 0.9)**: System catches most defects in each category
- **Medium Recall (0.7-0.9)**: Catches majority but misses some
- **Low Recall (< 0.7)**: Misses many defects - significant gaps in coverage

**When to Use**:
- When missing defects is costly (e.g., critical security issues)
- When you want comprehensive coverage
- When false negatives are more problematic than false positives

**Real-World Impact**:
- **High Recall**: Comprehensive defect detection, few slip through
- **Low Recall**: Many defects go undetected or miscategorized

**Example**:
```
Actual database defects: 25
Correctly identified: 20
Missed (classified as other): 5
Recall = 20/(20+5) = 0.80 (80%)
```

---

### 4. F1 Score

**Definition**: Harmonic mean of precision and recall, providing a single balanced metric.

**Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Range**: 0.0 to 1.0 (0% to 100%)

**Calculation Method**:
```python
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
```

**Why Harmonic Mean?**
- Punishes extreme imbalances between precision and recall
- Regular average would hide poor performance in one metric
- Example: Precision=1.0, Recall=0.1
  - Arithmetic mean: 0.55 (misleading)
  - Harmonic mean (F1): 0.18 (reveals the issue)

**Interpretation**:
- **F1 > 0.9**: Excellent balance between precision and recall
- **F1 0.7-0.9**: Good balance, acceptable for production
- **F1 0.5-0.7**: Moderate performance, needs improvement
- **F1 < 0.5**: Poor performance, significant issues

**When to Use**:
- When you need to balance precision and recall
- When comparing different models/configurations
- When no clear priority between false positives and false negatives
- As the primary metric for model optimization

**Trade-offs**:
| Scenario | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Conservative (few predictions) | High | Low | Medium |
| Aggressive (many predictions) | Low | High | Medium |
| Balanced | Medium | Medium | High |

**Example**:
```
Precision = 0.85
Recall = 0.80
F1 = 2 × (0.85 × 0.80) / (0.85 + 0.80) = 0.824 (82.4%)
```

---

## Confidence Metrics

### 5. Average Confidence

**Definition**: The mean confidence score across all predictions, indicating the agent's certainty in its classifications.

**Formula**:
```
Avg Confidence = Σ(confidence_scores) / n
```

**Range**: 0.0 to 1.0 (0% to 100%)

**Calculation Method**:
```python
avg_confidence = np.mean([result.confidence for result in results])
```

**Interpretation**:
- **High Average (> 0.8)**: Agent is generally confident in predictions
- **Medium Average (0.5-0.8)**: Moderate uncertainty
- **Low Average (< 0.5)**: Agent frequently uncertain

**Ideal Patterns**:
- **High confidence + High accuracy**: System is both confident and correct
- **High confidence + Low accuracy**: Overconfident - dangerous pattern
- **Low confidence + Low accuracy**: System knows it's struggling
- **Low confidence + High accuracy**: Under-confident - opportunity for tuning

**When to Use**:
- Assessing overall system certainty
- Setting confidence thresholds for automated actions
- Identifying cases that need human review
- Monitoring model degradation over time

**Actionable Thresholds**:
```python
if confidence >= 0.9:
    # Automatic categorization
elif confidence >= 0.7:
    # Semi-automatic with review
else:
    # Manual review required
```

---

### 6. Confidence Calibration

**Definition**: Correlation between the agent's confidence scores and actual correctness, measuring how well confidence reflects accuracy.

**Formula**:
```
Confidence Calibration = Pearson Correlation(confidence_scores, correctness_binary)
```

**Range**: -1.0 to 1.0

**Calculation Method**:
```python
confidences = np.array([r.confidence for r in results])
correctness = np.array([1.0 if r.is_correct else 0.0 for r in results])
confidence_calibration = np.corrcoef(confidences, correctness)[0, 1]
```

**Interpretation**:
- **+1.0**: Perfect calibration - higher confidence always means correct
- **+0.7 to +1.0**: Well calibrated - confidence is reliable
- **+0.3 to +0.7**: Moderately calibrated - some relationship
- **0.0**: No relationship - confidence is meaningless
- **Negative**: Inverse relationship - high confidence predicts errors (very bad!)

**Calibration Quality Examples**:

| Calibration | Meaning | Example |
|-------------|---------|---------|
| 0.95 | Excellent | 90% confident → 90% correct |
| 0.70 | Good | 80% confident → 75% correct |
| 0.30 | Poor | 80% confident → 60% correct |
| 0.00 | Uncalibrated | Confidence unrelated to accuracy |
| -0.50 | Inverse | High confidence → likely wrong! |

**Why It Matters**:
- Determines if confidence scores can be trusted for decision-making
- Essential for automated triage systems
- Affects user trust in the system
- Guides threshold setting for automated vs manual review

**Improving Calibration**:
1. **More Training Data**: Especially for edge cases
2. **Temperature Scaling**: Adjust confidence scores post-hoc
3. **Ensemble Methods**: Combine multiple models
4. **Regular Retraining**: Adapt to changing patterns

**Visual Interpretation**:
```
Well Calibrated:
Confidence: [0.9, 0.8, 0.7, 0.5]
Correct:    [Yes, Yes, No,  No]
Pattern: Higher confidence = More likely correct

Poorly Calibrated:
Confidence: [0.9, 0.8, 0.7, 0.5]
Correct:    [No,  Yes, Yes, Yes]
Pattern: No clear relationship
```

---

## Performance Metrics

### 7. Average Response Time

**Definition**: Mean time taken to analyze a single defect, from input to result.

**Unit**: Seconds (float)

**Calculation Method**:
```python
avg_response_time = np.mean([result.response_time for result in results])
```

**Typical Values**:
- **< 1 second**: Excellent - real-time analysis
- **1-5 seconds**: Good - acceptable for most use cases
- **5-10 seconds**: Acceptable - may need optimization
- **> 10 seconds**: Slow - investigate bottlenecks

**Components of Response Time**:
1. **Vector Search** (~20-30%): Finding similar historical defects
2. **LLM Processing** (~60-70%): Azure OpenAI analysis
3. **Pre/Post Processing** (~5-10%): Parsing, formatting

**Factors Affecting Performance**:
- **Vector Database Size**: More documents = slower search (use IVF/HNSW)
- **LLM Model**: GPT-4 slower than GPT-3.5
- **Token Count**: Longer contexts increase processing time
- **top_k Parameter**: More similar defects = more processing
- **Network Latency**: Azure OpenAI API round-trip time

**Optimization Strategies**:
```python
# Fast configuration (sacrifices some accuracy)
agent = DefectTriageAgent(
    ...,
    top_k=3,           # Fewer similar defects
    max_tokens=800,    # Shorter responses
    temperature=0.1    # More deterministic
)

# Balanced configuration
agent = DefectTriageAgent(
    ...,
    top_k=5,
    max_tokens=1500,
    temperature=0.3
)
```

**Performance Monitoring**:
```python
# Log slow queries
if response_time > 5.0:
    logger.warning(f"Slow analysis: {response_time:.2f}s for case {case_id}")
    
# Track percentiles
p50 = np.percentile(response_times, 50)  # Median
p95 = np.percentile(response_times, 95)  # 95th percentile
p99 = np.percentile(response_times, 99)  # 99th percentile
```

**SLA Considerations**:
- **Interactive UI**: Target < 3 seconds (p95)
- **Batch Processing**: 5-10 seconds acceptable
- **Critical Alerts**: May need < 1 second cache-based system

---

## Category-Specific Metrics

### 8. Category Accuracy

**Definition**: Accuracy computed separately for each defect category (database, memory, null_pointer, etc.).

**Formula**:
```
Category Accuracy = Correct Predictions in Category / Total Cases in Category
```

**Storage**:
```python
category_accuracy: Dict[str, float] = {
    "database": 0.92,
    "memory": 0.85,
    "null_pointer": 0.88,
    "network": 0.75,
    "configuration": 0.80
}
```

**Why Track Per-Category?**
- Overall accuracy can mask category-specific problems
- Some categories may be harder to classify
- Helps identify where to focus improvement efforts
- Reveals data imbalance issues

**Analysis Examples**:

**Scenario 1: Balanced Performance**
```python
{
    "database": 0.85,
    "memory": 0.83,
    "null_pointer": 0.87,
    "network": 0.84
}
# Interpretation: Consistent performance across categories
```

**Scenario 2: Imbalanced Performance**
```python
{
    "database": 0.95,      # Very good
    "memory": 0.90,        # Good
    "null_pointer": 0.60,  # Poor - needs attention!
    "network": 0.55        # Poor - needs attention!
}
# Action: Add more training examples for null_pointer and network
```

**Scenario 3: Rare Category Problem**
```python
category_counts = {
    "database": 45,        # Well represented
    "memory": 38,          # Well represented
    "null_pointer": 12,    # Under-represented
    "network": 5           # Very under-represented
}
category_accuracy = {
    "database": 0.91,
    "memory": 0.89,
    "null_pointer": 0.75,  # Worse due to fewer examples
    "network": 0.60        # Much worse - need more data
}
# Action: Collect more examples for rare categories
```

**Minimum Sample Sizes**:
- **< 10 samples**: Unreliable - need more data
- **10-30 samples**: Indicative but noisy
- **30-100 samples**: Reasonably reliable
- **> 100 samples**: Statistically robust

---

### 9. Category Counts

**Definition**: Number of test cases in each category, revealing dataset composition.

**Purpose**:
- Identify dataset imbalance
- Validate statistical significance of per-category metrics
- Guide data collection priorities
- Assess category distribution

**Example Analysis**:
```python
category_counts = {
    "database": 45,
    "memory": 12,
    "null_pointer": 8,
    "network": 3,
    "configuration": 2
}

# Calculate distribution
total = sum(category_counts.values())  # 70
for category, count in category_counts.items():
    percentage = (count / total) * 100
    print(f"{category}: {count} ({percentage:.1f}%)")

# Output:
# database: 45 (64.3%)        <- Dominant category
# memory: 12 (17.1%)
# null_pointer: 8 (11.4%)
# network: 3 (4.3%)            <- Under-represented
# configuration: 2 (2.9%)      <- Severely under-represented
```

**Implications of Imbalance**:
1. **Overall Accuracy Bias**: High accuracy on dominant categories can inflate overall score
2. **Learning Bias**: Model may default to predicting common categories
3. **Rare Category Errors**: Under-represented categories will have lower accuracy
4. **Metric Interpretation**: Weighted metrics become crucial

**Mitigation Strategies**:
1. **Collect More Data**: Actively seek rare category examples
2. **Use Weighted Metrics**: precision/recall with `average='weighted'`
3. **Stratified Sampling**: Ensure balanced test sets
4. **Class Weights**: If training custom classifiers

---

## Interpretation Guidelines

### Holistic Metric Analysis

Don't evaluate metrics in isolation. Consider the complete picture:

#### Pattern 1: High-Confidence Correct System
```
Accuracy: 0.92
Precision: 0.90
Recall: 0.91
F1 Score: 0.905
Avg Confidence: 0.88
Confidence Calibration: 0.85
```
**Interpretation**: Excellent system ready for production. High accuracy with well-calibrated confidence scores. Can trust automated decisions.

#### Pattern 2: Overconfident System
```
Accuracy: 0.65
Precision: 0.63
Recall: 0.67
F1 Score: 0.65
Avg Confidence: 0.92
Confidence Calibration: 0.10
```
**Interpretation**: Dangerous! System is very confident but often wrong. Confidence scores cannot be trusted. Needs recalibration or more training data.

#### Pattern 3: Underconfident System
```
Accuracy: 0.88
Precision: 0.87
Recall: 0.89
F1 Score: 0.88
Avg Confidence: 0.55
Confidence Calibration: 0.75
```
**Interpretation**: Good accuracy but low confidence. May be overly cautious. Could optimize thresholds to enable more automated decisions.

#### Pattern 4: Imbalanced Specialist
```
Overall Accuracy: 0.85
Category Accuracy:
  - database: 0.95
  - memory: 0.90
  - null_pointer: 0.60
  - network: 0.55
```
**Interpretation**: Excellent at common categories but struggles with rare ones. Need more diverse training data.

### Performance Baselines

Compare against these baselines to assess system quality:

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| Accuracy | 0.60 | 0.80 | 0.90 |
| Precision | 0.60 | 0.80 | 0.90 |
| Recall | 0.60 | 0.80 | 0.90 |
| F1 Score | 0.60 | 0.80 | 0.90 |
| Confidence Calibration | 0.30 | 0.60 | 0.80 |
| Response Time | 10s | 3s | 1s |

### Red Flags

Watch for these warning signs:

1. **Accuracy >> F1**: Likely dataset imbalance
2. **High Precision + Low Recall**: Too conservative, missing many defects
3. **Low Precision + High Recall**: Too aggressive, many false alarms
4. **Low Confidence Calibration**: Can't trust confidence scores
5. **Negative Confidence Calibration**: System is completely miscalibrated
6. **Large Category Variance**: Some categories perform much worse
7. **Response Time > 10s**: Performance bottleneck

---

## Best Practices

### 1. Continuous Monitoring

```python
# Track metrics over time
metrics_history = []

for evaluation_run in evaluation_runs:
    metrics = evaluate_model(test_set)
    metrics_history.append({
        'timestamp': datetime.now(),
        'metrics': metrics
    })
    
    # Alert on degradation
    if metrics.accuracy < previous_accuracy - 0.05:
        alert("Model accuracy degraded by >5%")
```

### 2. Stratified Evaluation

```python
# Ensure balanced test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    stratify=labels  # Maintain class distribution
)
```

### 3. Multiple Metrics

Always report multiple metrics:
```python
report = f"""
Evaluation Results:
==================
Accuracy:     {metrics.accuracy:.2%}
Precision:    {metrics.precision:.2%}
Recall:       {metrics.recall:.2%}
F1 Score:     {metrics.f1_score:.2%}
Calibration:  {metrics.confidence_calibration:.3f}
Avg Time:     {metrics.avg_response_time:.2f}s

Per-Category Performance:
{format_category_metrics(metrics)}
"""
```

### 4. A/B Testing

When comparing configurations:
```python
# Statistical significance testing
from scipy import stats

results_a = [0.85, 0.87, 0.86, 0.84, 0.88]
results_b = [0.90, 0.91, 0.89, 0.92, 0.90]

t_stat, p_value = stats.ttest_ind(results_a, results_b)

if p_value < 0.05:
    print("Configuration B is significantly better")
else:
    print("No significant difference")
```

### 5. Error Analysis

Beyond metrics, analyze errors:
```python
# Find patterns in errors
errors = [r for r in results if not r.is_correct]

# Group by error type
error_patterns = {}
for error in errors:
    pattern = f"{error.ground_truth_category} → {error.predicted_category}"
    error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

# Most common errors
for pattern, count in sorted(error_patterns.items(), 
                            key=lambda x: x[1], reverse=True):
    print(f"{pattern}: {count} errors")
```

### 6. Production Monitoring

```python
# Real-time metric tracking
class MetricsTracker:
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0
        self.response_times = []
        
    def record_prediction(self, is_correct, response_time):
        self.total_count += 1
        if is_correct:
            self.correct_count += 1
        self.response_times.append(response_time)
        
        # Alert on degradation
        if self.total_count > 100:
            current_accuracy = self.correct_count / self.total_count
            if current_accuracy < 0.75:
                self.alert("Accuracy dropped below 75%")
```

---

## Summary

### Quick Reference

| Metric | What It Measures | Target | Red Flag |
|--------|------------------|--------|----------|
| **Accuracy** | Overall correctness | > 0.80 | < 0.70 |
| **Precision** | Prediction reliability | > 0.80 | < 0.70 |
| **Recall** | Coverage completeness | > 0.80 | < 0.70 |
| **F1 Score** | Balanced performance | > 0.80 | < 0.70 |
| **Avg Confidence** | System certainty | 0.70-0.90 | < 0.50 or > 0.95 |
| **Calibration** | Confidence reliability | > 0.60 | < 0.30 or negative |
| **Response Time** | Processing speed | < 3s | > 10s |
| **Category Accuracy** | Category-specific performance | Similar across categories | High variance |

### Key Takeaways

1. **No Single Metric**: Always evaluate multiple metrics together
2. **Context Matters**: Interpret metrics based on use case requirements
3. **Balance is Key**: F1 score helps balance precision and recall
4. **Trust Calibration**: Confidence calibration is crucial for automated systems
5. **Monitor Trends**: Track metrics over time, not just point-in-time
6. **Category Analysis**: Per-category metrics reveal hidden issues
7. **Performance Counts**: Response time affects user experience
8. **Statistical Significance**: Ensure adequate sample sizes

---

**Version**: 1.0.0  
**Last Updated**: January 12, 2026  
**Related**: [evaluation.py](../evaluation/evaluation.py), [README.md](../README.md)
