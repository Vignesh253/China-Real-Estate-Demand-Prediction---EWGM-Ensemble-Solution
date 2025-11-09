# China-Real-Estate-Demand-Prediction---EWGM-Ensemble-Solution

# China Real Estate Demand Prediction - Competition Solution

##  Solution Overview

This solution achieves high accuracy through an advanced **Exponentially Weighted Geometric Mean (EWGM)** ensemble combined with sector-specific **December seasonality adjustment**. The approach leverages temporal patterns and sector behaviors to predict future housing transaction volumes.

### Key Innovation: Hybrid EWGM + Last Value Predictor

Our winning formula combines two powerful components:
- **34% Last Value**: Captures recent momentum and market continuity
- **34% EWGM (12-month window)**: Weighted geometric mean that emphasizes recent trends while incorporating historical patterns
- **December Multiplier**: Sector-specific adjustment (0.85–1.40×) to capture end-of-year market dynamics

**Why This Works:**
1. **Geometric Mean** handles skewed distributions better than arithmetic mean (common in real estate data)
2. **Exponential Weighting** (α=0.5) gives more importance to recent months while still considering long-term patterns
3. **Dead Sector Detection** prevents overfitting on inactive markets
4. **Seasonality Adjustment** captures December buying patterns specific to each sector

---

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Requirements](#data-requirements)
3. [Reproducing Results](#reproducing-results)
4. [Methodology](#methodology)
5. [Configuration & Hyperparameters](#configuration--hyperparameters)
6. [Performance Metrics](#performance-metrics)
7. [Compliance & Originality](#compliance--originality)

---

## 🔧 Environment Setup

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), Windows 10+, or macOS 11+
- **Python**: 3.11.13 (tested and recommended)
- **RAM**: Minimum 4GB, Recommended 8GB+
- **GPU**: Optional (NVIDIA Tesla T4 or equivalent) - used for Kaggle Notebook execution
- **Disk Space**: ~500MB for environment + data

### Installation

#### Option 1: Using requirements.txt (Recommended)

