# IFN647-IR

# Information Retrieval System

A comprehensive implementation of three information retrieval models: **BM25**, **Language Model with Jelinek-Mercer smoothing (LMRM)**, and **Pseudo-Relevance Ranking Model (PRRM)** with complete evaluation and statistical analysis.

## Features

- **BM25 Implementation**: Classic probabilistic ranking function
- **LMRM Implementation**: Language model with Jelinek-Mercer smoothing (λ=0.4)
- **PRRM Implementation**: Machine learning model using pseudo-relevance feedback
- **Comprehensive Evaluation**: AP, P@12, DCG@12 metrics for all models
- **Statistical Analysis**: 9 pairwise t-tests with effect size calculations
- **Automated Pipeline**: One-command execution of entire system
- **Detailed Logging**: Progress tracking and top-12 document display

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)

## Installation

### Prerequisites

- Python 3.7 or higher

### Install Dependencies

```bash
pip install numpy scipy pandas scikit-learn
```

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/information-retrieval-system.git
   cd information-retrieval-system
   ```

2. **Set up your data** (see [Data Setup](#data-setup) for details):
   ```
   data/
   ├── Queries-1.txt
   ├── common-english-words.txt
   ├── DataSets/
   └── EvaluationBenchmark/
   ```

3. **Run the complete pipeline**:
   ```bash
   python main.py
   ```

4. **Check results**:
   ```
   outputs/
   ├── BM25/rankings/
   ├── LMRM/rankings/
   └── PRRM/rankings/
   ```

## Project Structure

```
├── main.py                     # Main orchestrator script
├── src/                        # Source code
│   ├── BM25IR.py              # BM25 implementation
│   ├── LMRM.py                # Language model implementation
│   ├── PRRM.py                # Pseudo-relevance model
│   ├── run_bm25.py            # BM25 execution script
│   ├── run_lmrm.py            # LMRM execution script
│   ├── run_prrm.py            # PRRM execution script
│   ├── statistical_analysis.py # Comprehensive t-tests
│   ├── evaluation_*.py         # Evaluation modules
│   ├── data_processing_*.py    # Data processing modules
│   ├── feature_extraction_*.py # Feature extraction
│   └── stemming.py            # Porter2 stemmer
├── data/                       # Input data (user-provided)
│   ├── Queries-1.txt
│   ├── common-english-words.txt
│   ├── DataSets/
│   └── EvaluationBenchmark/
├── outputs/                    # Generated results
│   ├── BM25/
│   ├── LMRM/
│   └── PRRM/
├── README.md
└── requirements.txt
```

## Data Setup

### Required Files

1. **Queries-1.txt**: Query definitions in TREC format
   ```xml
   <Query>
   <num> Number: R101
   <title> Your query title
   <desc> Description: Detailed description
   <narr> Narrative: Detailed narrative
   </Query>
   ```

2. **common-english-words.txt**: Comma-separated stop words
   ```
   a,able,about,across,after,all,almost,also,am,among,an,and,any...
   ```

3. **DataSets/**: Document collections
   ```
   DataSets/
   ├── Dataset101/     # XML files for query R101
   ├── Dataset102/     # XML files for query R102
   └── ...
   ```

4. **EvaluationBenchmark/**: Relevance judgments
   ```
   Dataset101.txt:
   R101 12345 1
   R101 67890 0
   R101 54321 1
   ```

### XML Document Format

```xml
<newsitem itemid="12345">
<text>
<p>Document content here</p>
<p>More content here</p>
</text>
</newsitem>
```

## Usage

### Complete Pipeline

```bash
python main.py
```

Runs all models in sequence and generates comprehensive results.

### Individual Models

```bash
cd src

# Run BM25 only
python run_bm25.py

# Run LMRM only
python run_lmrm.py

# Run PRRM only (requires BM25 and LMRM outputs)
python run_prrm.py

# Run statistical analysis
python statistical_analysis.py
```

### Custom Configuration

Modify parameters in source files:

```python
# BM25IR.py
k1 = 1.2  # Term frequency saturation
b = 0.75  # Document length normalization

# LMRM.py
LAMBDA_VAL = 0.4  # Jelinek-Mercer smoothing

# PRRM.py
C = 1.0  # Regularization parameter
```

## Models

### BM25 (Best Matching 25)

**Implementation**: `BM25IR.py`

Classic probabilistic ranking function with:
- k₁ = 1.2 (term frequency saturation)
- b = 0.75 (document length normalization)
- Handles negative scores with 3×N adjustment

**Formula**:
```
BM25(d,q) = Σ IDF(qᵢ) × (f(qᵢ,d) × (k₁ + 1)) / (f(qᵢ,d) + k₁ × (1 - b + b × |d|/avgdl))
```

### LMRM (Language Model with Jelinek-Mercer)

**Implementation**: `LMRM.py`

Statistical language model with:
- Jelinek-Mercer smoothing (λ = 0.4)
- Log-probability scoring
- Porter2 stemming

**Formula**:
```
P(q|d) = Π [(1-λ) × P(qᵢ|d) + λ × P(qᵢ|C)]
Score = log₂(P(q|d))
```

### PRRM (Pseudo-Relevance Ranking Model)

**Implementation**: `PRRM.py`

Machine learning approach using:
- Pseudo-relevance feedback from BM25 and LMRM
- Logistic regression with L2 regularization
- TF-IDF cosine similarity features
- Balanced class weights

**Features**:
- BM25 and LMRM scores
- Cosine similarity with query
- Document and query interaction features

## Evaluation

### Metrics

- **MAP (Mean Average Precision)**: Overall ranking quality
- **P@12 (Precision at 12)**: Precision in top 12 results
- **DCG@12 (Discounted Cumulative Gain at 12)**: Ranked relevance quality


### Output Files

```
outputs/
├── BM25/
│   ├── rankings/BM25IR_R101Ranking.dat
│   └── evaluations/BM25_AP.csv
├── LMRM/
│   ├── rankings/LMRM_R101Ranking.dat
│   └── LMRM_Evaluation_Results.csv
└── PRRM/
    ├── rankings/PRRM_R101Ranking.dat
    └── PRRM_Evaluation_Results.csv
```

## Example Use Cases

### Academic Research
- Information retrieval experiments
- Baseline model comparisons
- Method validation studies

### Industry Applications
- Document recommendation systems
- Content ranking algorithms

## References

### Algorithms
- **BM25**: Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
- **Language Models**: Jelinek, F., & Mercer, R. L. (1980). Interpolated estimation of Markov source parameters.


### Implementation
- **Porter Stemming**: Porter, M. F. (2001). Snowball: A language for stemming algorithms.
- **Evaluation Metrics**: Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
