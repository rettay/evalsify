# P<designation>™/D1

# Draft Standard for Evaluating LLM-Powered Applications, Systems, and Products

Developed by the  
**Artificial Intelligence Standards Committee (AISC)**  
of the  
**IEEE Computer Society** 

Approved <Date TBD>

**IEEE SA Standards Board**

Copyright © 2025 by The Institute of Electrical and Electronics Engineers, Inc.  
Three Park Avenue  
New York, New York 10016-5997, USA

All rights reserved.

**Abstract:** This standard specifies a comprehensive, interoperable framework for evaluating LLM-powered applications and systems, incorporating safety, security, and bias considerations. Grounded in the atomic unit of evaluation—a single model-task assessment consisting of an input, expected output ("Golden"), rubric, and score—the standard defines JSON-based schemas, statistical validation requirements, conformance testing procedures, and quality assurance protocols to enable consistent, reproducible, and statistically valid assessments across tools and organizations.

**Keywords:** AI evaluation, golden datasets, evaluation rubrics, statistical validation, conformance testing, safety evaluation, bias detection, interoperability, JSON schema, model benchmarking

---

**Contents**

1. Overview  
   1.1 Scope  
   1.2 Purpose  
   1.3 Word Usage  
2. Normative References  
3. Definitions, Acronyms, and Abbreviations  
   3.1 Definitions  
   3.2 Acronyms and Abbreviations  
4. Evaluation Framework Components  
   4.1 General  
   4.2 Evaluation Datasets  
   4.3 Evaluation Rubrics  
   4.4 Evaluation Specifications  
   4.5 Evaluation Aggregation  
5. Quality Assurance Requirements  
   5.1 General  
   5.2 Dataset Quality Requirements  
   5.3 Statistical Validation Requirements  
   5.4 Version Control and Provenance  
6. Safety, Security, and Bias Considerations  
   6.1 General  
   6.2 Safety Evaluation Requirements  
   6.3 Security Assessment Protocols  
   6.4 Bias Detection and Mitigation  
   6.5 Adversarial Testing Requirements  
7. JSON Schema Specifications  
   7.1 General  
   7.2 Dataset Schema  
   7.3 Rubric Schema  
   7.4 Evaluation Specification Schema  
   7.5 Schema Versioning and Migration  
8. Conformance and Testing  
   8.1 General  
   8.2 Conformance Requirements  
   8.3 Interoperability Testing  
   8.4 Certification Procedures  
9. Use Cases and Integration  
   9.1 General  
   9.2 Multi-Modal Evaluations  
   9.3 Agentic AI Systems  
   9.4 Real-Time and Streaming Evaluation  
   9.5 Federated Evaluation Approaches  

Annex A (informative) Bibliography  
Annex B (informative) Example Implementation  
Annex C (normative) Conformance Test Procedures  
Annex D (informative) Statistical Methods for Evaluation  
Annex E (informative) Security Considerations Checklist  
Annex F (informative) Multi-Modal and Agentic Extensions

---

## 1. Overview

### 1.1 Scope

This standard specifies a comprehensive framework for evaluating LLM-powered applications and systems, anchored in single model-task evaluations as the atomic unit. Each evaluation unit consists of an input, expected output ("Golden"), rubric, and resulting score with associated confidence intervals and statistical validation requirements.

The standard defines:

- **Evaluation Datasets:** Collections of atomic evaluations serving as ground truth benchmarks with quality assurance protocols
- **Evaluation Rubrics:** Metrics and criteria for assessing model outputs with statistical validation requirements
- **Evaluation Specifications:** Configurations combining datasets and rubrics with conformance testing procedures
- **Safety and Security Protocols:** Requirements for bias detection, adversarial testing, and ethical evaluation
- **Quality Assurance Framework:** Statistical validation, version control, and provenance tracking
- **Conformance Testing:** Certification procedures and interoperability validation

The framework supports single-turn, multi-turn, multi-modal, and agentic AI evaluations, ensuring extensibility for emerging AI capabilities while maintaining statistical rigor and reproducibility.

### 1.2 Purpose

This standard establishes a comprehensive, statistically sound framework for AI model evaluation that addresses critical gaps in existing evaluation methodologies. It provides:

- **Technical Rigor:** Statistical validation requirements, confidence intervals, and mathematical formulations comparable to established IEEE AI standards
- **Safety and Security:** Mandatory bias detection, adversarial testing, and ethical evaluation protocols
- **Interoperability:** Machine-readable schemas enabling cross-tool compatibility and reproducible research
- **Quality Assurance:** Data provenance tracking, version control, and validation procedures
- **Conformance:** Clear certification requirements and testing protocols

The standard enables AI developers, researchers, and organizations to conduct scientifically valid evaluations while ensuring safety, fairness, and reliability of AI systems.

### 1.3 Word Usage

The word **shall** indicates mandatory requirements strictly to be followed in order to conform to the standard and from which no deviation is permitted.

The word **should** indicates recommendations that are particularly suitable without excluding alternatives.

The word **may** indicates permissible courses of action within the standard's limits.

The word **can** indicates statements of possibility and capability.

## 2. Normative References

The following referenced documents are indispensable for the application of this document:

- **IEEE 2802-2022,** *IEEE Standard for Performance and Safety Evaluation of Artificial Intelligence Based Medical Devices: Terminology*
- **IEEE 2894-2024,** *IEEE Guide for an Architectural Framework for Explainable Artificial Intelligence*
- **ISO/IEC 23053:2022,** *Information Technology — Artificial Intelligence — Framework for AI systems using machine learning*
- **ISO/IEC 23894:2023,** *Information Technology — Artificial Intelligence — Guidance on risk management*
- **ISO/IEC 42001:2023,** *Information Technology — Artificial Intelligence — Management system*
- **JSON Schema Core,** *Internet Engineering Task Force (IETF) Draft, draft-bhutton-json-schema-01, June 2022*
- **RFC 2119,** *Key words for use in RFCs to Indicate Requirement Levels, S. Bradner, March 1997*
- **RFC 8259,** *The JavaScript Object Notation (JSON) Data Interchange Format, T. Bray, December 2017*

## 3. Definitions, Acronyms, and Abbreviations

### 3.1 Definitions

**Adversarial Testing:** Systematic evaluation of AI systems using inputs designed to reveal vulnerabilities, biases, or failure modes.

**Agentic AI System:** AI system capable of autonomous decision-making, planning, and multi-step task execution with environmental interaction.

**Atomic Evaluation Unit:** Single model-task evaluation consisting of an input, expected output (Golden), rubric, and resulting score with confidence metrics. Foundation unit for all evaluation constructs.

**Confidence Interval:** Statistical range providing bounds on population parameters with specified probability level, typically 95%.

**Evaluation Dataset:** Collection of atomic evaluation units serving as ground truth benchmarks with metadata, quality metrics, and provenance information.

**Evaluation Rubric:** Set of criteria, metrics, and statistical validation requirements for assessing AI outputs against expected results.

**Evaluation Specification:** Configuration defining complete evaluation workflows by combining datasets, rubrics, and execution parameters with conformance requirements.

**Ground Truth:** Verified correct or ideal output for given input, serving as reference standard for evaluation with documented validation process.

**Multi-Modal Evaluation:** Assessment involving multiple input/output modalities (text, image, audio, video) within single evaluation framework.

**Statistical Validation:** Process ensuring evaluation results meet statistical significance requirements with appropriate confidence intervals and hypothesis testing.

### 3.2 Acronyms and Abbreviations

**AI:** Artificial Intelligence  
**API:** Application Programming Interface  
**CI:** Confidence Interval  
**JSON:** JavaScript Object Notation  
**LLM:** Large Language Model  
**MANOVA:** Multivariate Analysis of Variance  
**MSE:** Mean Squared Error  
**ROC:** Receiver Operating Characteristic

## 4. Evaluation Framework Components

### 4.1 General

This standard defines four core components with enhanced technical requirements: Evaluation Datasets, Evaluation Rubrics, Evaluation Specifications, and Quality Assurance Protocols. Each component shall be represented in validated JSON schemas that are human-readable, machine-processable, statistically sound, and interoperable across evaluation tools.

### 4.2 Evaluation Datasets

Evaluation Datasets shall consist of collections of atomic evaluation units with comprehensive metadata and quality assurance protocols.

#### 4.2.1 Dataset Requirements

Each dataset shall include:

- **Unique Identifier:** UUID or namespace-qualified identifier
- **Versioning:** Semantic versioning (MAJOR.MINOR.PATCH) with migration procedures
- **Quality Metrics:** Inter-annotator agreement scores, confidence intervals for ground truth
- **Provenance:** Data source documentation, collection methodology, validation procedures
- **Statistical Properties:** Sample size justification, power analysis, distribution characteristics

#### 4.2.2 Atomic Unit Specifications

Each atomic evaluation unit shall contain:

- **Input:** String, structured JSON object, or multi-modal reference with metadata
- **Expected Output:** String, structured object, or multiple acceptable responses with confidence scores
- **Validation Status:** Quality check results, reviewer consensus scores
- **Metadata:** Category, difficulty level, demographic tags, bias indicators

#### 4.2.3 Quality Assurance

Datasets shall undergo validation including:

- **Statistical Adequacy:** Minimum sample size calculations based on effect size and power requirements
- **Bias Assessment:** Demographic representation analysis, fairness metrics evaluation
- **Inter-Annotator Reliability:** Cohen's kappa ≥ 0.70 for categorical labels, intraclass correlation ≥ 0.80 for continuous measures

### 4.3 Evaluation Rubrics

Evaluation Rubrics shall define statistically validated criteria for assessing AI outputs with confidence intervals and significance testing.

#### 4.3.1 Rubric Requirements

Each rubric shall specify:

- **Metric Definition:** Mathematical formulation with assumptions and constraints
- **Statistical Validation:** Significance testing procedures, confidence interval calculations
- **Calibration Requirements:** Methods for ensuring measurement validity and reliability
- **Error Analysis:** Procedures for identifying and categorizing evaluation errors

#### 4.3.2 Supported Metrics

Standard metric types shall include:

- **Exact Match:** Binary comparison with case sensitivity and normalization options
- **Semantic Similarity:** Embedding-based metrics with confidence intervals
- **Statistical Measures:** Precision, recall, F1-score with bootstrap confidence intervals
- **Model-based Judging:** LLM evaluation with reliability assessment and bias detection

#### 4.3.3 Statistical Requirements

All rubrics shall provide:

- **Confidence Intervals:** 95% confidence bounds using appropriate statistical methods
- **Significance Testing:** Hypothesis testing procedures with Type I/II error control
- **Effect Size Metrics:** Cohen's d, eta-squared, or appropriate measures
- **Reliability Metrics:** Test-retest reliability, internal consistency measures

### 4.4 Evaluation Specifications

Evaluation Specifications shall define complete, reproducible evaluation workflows with statistical validation and conformance requirements.

#### 4.4.1 Specification Components

Each specification shall include:

- **Dataset References:** Validated dataset identifiers or embedded objects
- **Rubric Configurations:** Statistical parameters and validation requirements
- **Execution Parameters:** Model settings, sampling procedures, randomization controls
- **Reporting Requirements:** Mandatory metrics, confidence intervals, effect sizes

#### 4.4.2 Reproducibility Requirements

Specifications shall ensure reproducibility through:

- **Randomization Control:** Fixed seeds, stratified sampling procedures
- **Environmental Specification:** Software versions, hardware requirements, configuration parameters
- **Data Partitioning:** Training/validation/test split procedures with cross-validation protocols

### 4.5 Evaluation Aggregation

#### 4.5.1 Statistical Aggregation

Evaluation results shall be aggregated using statistically sound methods:

- **Score Combination:** Weighted averages with confidence interval propagation
- **Multi-Rubric Integration:** MANOVA or appropriate multivariate methods
- **Uncertainty Quantification:** Bootstrap confidence intervals, Bayesian credible intervals

#### 4.5.2 Reporting Requirements

Aggregated results shall include:

- **Point Estimates:** Mean scores with appropriate central tendency measures
- **Confidence Intervals:** 95% bounds using validated statistical methods
- **Effect Sizes:** Practical significance measures beyond statistical significance
- **Uncertainty Metrics:** Standard errors, confidence interval widths

## 5. Quality Assurance Requirements

### 5.1 General

This standard establishes comprehensive quality assurance protocols ensuring evaluation reliability, validity, and statistical soundness comparable to established scientific standards.

### 5.2 Dataset Quality Requirements

#### 5.2.1 Data Quality Metrics

Datasets shall meet quality thresholds:

- **Completeness:** <5% missing values with imputation procedures documented
- **Consistency:** Inter-rater agreement κ ≥ 0.70 for categorical, ICC ≥ 0.80 for continuous
- **Accuracy:** Ground truth validation with multiple independent sources
- **Representativeness:** Population coverage analysis with bias assessment

#### 5.2.2 Quality Control Procedures

Implementation shall include:

- **Automated Validation:** Schema compliance, statistical outlier detection
- **Human Review:** Expert validation of samples with quality scoring
- **Continuous Monitoring:** Quality drift detection, periodic re-validation

### 5.3 Statistical Validation Requirements

#### 5.3.1 Sample Size Requirements

Evaluations shall meet statistical power requirements:

- **Power Analysis:** Minimum 80% power for detecting meaningful effect sizes
- **Effect Size Specifications:** Cohen's conventions or domain-specific standards
- **Multiple Comparison Corrections:** Bonferroni, FDR, or appropriate methods

#### 5.3.2 Statistical Testing

Implementations shall provide:

- **Hypothesis Testing:** Appropriate tests with assumption validation
- **Confidence Intervals:** Bootstrap, parametric, or non-parametric as appropriate
- **Significance Reporting:** p-values with effect sizes and practical significance

### 5.4 Version Control and Provenance

#### 5.4.1 Version Management

All evaluation assets shall maintain:

- **Semantic Versioning:** MAJOR.MINOR.PATCH with backward compatibility
- **Change Documentation:** Comprehensive logs of modifications with rationale
- **Migration Procedures:** Automated or documented upgrade paths

#### 5.4.2 Provenance Tracking

Implementation shall document:

- **Data Lineage:** Source documentation, transformation procedures
- **Evaluation History:** Complete audit trail of evaluations performed
- **Dependency Management:** Version tracking of all evaluation components

## 6. Safety, Security, and Bias Considerations

### 6.1 General

This standard mandates comprehensive safety, security, and fairness evaluation protocols addressing critical concerns in AI system deployment, following established frameworks from IEEE 2802-2022 and IEEE 2894-2024.

### 6.2 Safety Evaluation Requirements

#### 6.2.1 Safety Assessment Framework

Implementations shall conduct safety evaluations including:

- **Risk Identification:** Systematic hazard analysis using established taxonomies
- **Failure Mode Analysis:** FMEA or equivalent methodologies for identifying potential failures
- **Safety Metrics:** Quantitative measures of safety performance with confidence intervals
- **Harm Assessment:** Potential impact analysis with severity classifications

#### 6.2.2 Safety Testing Protocols

Safety validation shall include:

- **Boundary Testing:** Evaluation at operational limits and edge cases
- **Stress Testing:** Performance under extreme conditions or resource constraints
- **Degradation Analysis:** Graceful failure behavior assessment
- **Recovery Testing:** System restoration capabilities after failure

### 6.3 Security Assessment Protocols

#### 6.3.1 Security Requirements

Evaluation frameworks shall assess:

- **Input Validation:** Robustness against malformed or malicious inputs
- **Data Privacy:** PII detection and protection during evaluation
- **Model Extraction:** Resistance to model stealing attacks
- **Prompt Injection:** Resilience against adversarial prompt manipulation

#### 6.3.2 Security Testing Methodology

Security assessment shall employ:

- **Penetration Testing:** Systematic vulnerability assessment with standard tools
- **Red Team Exercises:** Adversarial evaluation by security experts
- **Threat Modeling:** Structured analysis of potential attack vectors
- **Compliance Verification:** Adherence to relevant security standards

### 6.4 Bias Detection and Mitigation

#### 6.4.1 Bias Assessment Requirements

Implementations shall evaluate multiple bias dimensions:

- **Demographic Parity:** Equal outcomes across protected groups with statistical testing
- **Equalized Odds:** Equal true positive/false positive rates across groups
- **Individual Fairness:** Similar outcomes for similar individuals with distance metrics
- **Intersectional Analysis:** Multi-dimensional bias assessment across group combinations

#### 6.4.2 Bias Testing Protocols

Bias evaluation shall include:

- **Fairness Metrics:** Quantitative measures with confidence intervals and significance tests
- **Subgroup Analysis:** Performance evaluation across demographic dimensions
- **Counterfactual Testing:** Analysis of decision changes under demographic modifications
- **Bias Auditing:** Systematic review using established bias detection frameworks

### 6.5 Adversarial Testing Requirements

#### 6.5.1 Adversarial Evaluation Framework

Systems shall undergo adversarial testing including:

- **Adversarial Examples:** Evaluation against crafted inputs designed to cause failures
- **Robustness Testing:** Performance assessment under various perturbation types
- **Attack Simulation:** Evaluation against known attack methodologies
- **Defense Validation:** Effectiveness of implemented countermeasures

#### 6.5.2 Adversarial Testing Protocols

Testing procedures shall employ:

- **Automated Attack Generation:** Systematic creation of adversarial examples
- **Human Red Teaming:** Expert-led adversarial evaluation
- **Robustness Metrics:** Quantitative measures of system resilience
- **Vulnerability Assessment:** Classification and prioritization of discovered weaknesses

## 7. JSON Schema Specifications

### 7.1 General

JSON schemas shall conform to JSON Schema Core specification with comprehensive validation rules, versioning support, and extension mechanisms for future capabilities.

### 7.2 Dataset Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://ieee.org/schemas/ai-evaluation/dataset/1.0",
  "title": "AI Evaluation Dataset Schema",
  "type": "object",
  "required": ["schema_version", "type", "id", "name", "examples", "quality_metrics"],
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "1.0",
      "description": "Schema version for compatibility tracking"
    },
    "type": {
      "type": "string",
      "const": "dataset"
    },
    "id": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "description": "Unique identifier following naming conventions"
    },
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 255
    },
    "description": {
      "type": "string",
      "maxLength": 2000
    },
    "version": {
      "type": "string",
      "pattern": "^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)$",
      "description": "Semantic versioning"
    },
    "license": {
      "type": "string",
      "description": "SPDX license identifier"
    },
    "author": {
      "oneOf": [
        {"type": "string"},
        {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "contact": {"type": "string", "format": "email"},
            "organization": {"type": "string"}
          },
          "required": ["name"]
        }
      ]
    },
    "quality_metrics": {
      "type": "object",
      "required": ["sample_size", "inter_annotator_agreement"],
      "properties": {
        "sample_size": {
          "type": "integer",
          "minimum": 1
        },
        "inter_annotator_agreement": {
          "type": "object",
          "properties": {
            "kappa": {"type": "number", "minimum": -1, "maximum": 1},
            "icc": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence_interval": {
              "type": "object",
              "properties": {
                "lower": {"type": "number"},
                "upper": {"type": "number"},
                "confidence_level": {"type": "number", "minimum": 0, "maximum": 1}
              }
            }
          }
        },
        "bias_assessment": {
          "type": "object",
          "properties": {
            "demographic_coverage": {"type": "object"},
            "fairness_metrics": {"type": "array"},
            "bias_score": {"type": "number", "minimum": 0, "maximum": 1}
          }
        }
      }
    },
    "provenance": {
      "type": "object",
      "properties": {
        "data_sources": {"type": "array", "items": {"type": "string"}},
        "collection_date": {"type": "string", "format": "date"},
        "validation_procedures": {"type": "array", "items": {"type": "string"}},
        "quality_controls": {"type": "array", "items": {"type": "string"}}
      }
    },
    "examples": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["input"],
        "properties": {
          "input": {
            "description": "Input data for evaluation"
          },
          "expected_output": {
            "description": "Expected output or acceptable responses"
          },
          "target_scores": {
            "type": "object",
            "description": "Scoring targets for multiple choice scenarios"
          },
          "confidence_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in ground truth accuracy"
          },
          "metadata": {
            "type": "object",
            "properties": {
              "category": {"type": "string"},
              "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
              "bias_indicators": {"type": "array", "items": {"type": "string"}},
              "validation_status": {"type": "string", "enum": ["validated", "pending", "rejected"]}
            }
          }
        }
      }
    }
  }
}
```

### 7.3 Rubric Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://ieee.org/schemas/ai-evaluation/rubric/1.0",
  "title": "AI Evaluation Rubric Schema",
  "type": "object",
  "required": ["schema_version", "type", "id", "name", "metric", "statistical_requirements"],
  "properties": {
    "schema_version": {"type": "string", "const": "1.0"},
    "type": {"type": "string", "const": "rubric"},
    "id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
    "name": {"type": "string", "minLength": 1, "maxLength": 255},
    "description": {"type": "string", "maxLength": 2000},
    "version": {
      "type": "string",
      "pattern": "^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)$"
    },
    "metric": {
      "type": "string",
      "enum": [
        "exact_match", "regex_match", "embedding_similarity", 
        "llm_judge", "statistical_test", "custom"
      ]
    },
    "statistical_requirements": {
      "type": "object",
      "required": ["confidence_level", "minimum_sample_size"],
      "properties": {
        "confidence_level": {
          "type": "number",
          "minimum": 0.8,
          "maximum": 0.99,
          "default": 0.95
        },
        "minimum_sample_size": {
          "type": "integer",
          "minimum": 30
        },
        "effect_size_threshold": {
          "type": "number",
          "minimum": 0.1,
          "description": "Minimum meaningful effect size"
        },
        "power_requirement": {
          "type": "number",
          "minimum": 0.8,
          "maximum": 1.0,
          "default": 0.8
        }
      }
    },
    "params": {
      "type": "object",
      "description": "Metric-specific parameters with validation"
    },
    "score_type": {
      "type": "string",
      "enum": ["binary", "continuous", "categorical", "ordinal"]
    },
    "calibration": {
      "type": "object",
      "properties": {
        "method": {"type": "string", "enum": ["platt_scaling", "isotonic_regression", "temperature_scaling"]},
        "parameters": {"type": "object"},
        "validation_score": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "bias_assessment": {
      "type": "object",
      "properties": {
        "fairness_constraints": {"type": "array", "items": {"type": "string"}},
        "subgroup_analysis": {"type": "boolean", "default": true},
        "bias_metrics": {"type": "array", "items": {"type": "string"}}
      }
    },
    "prompt_template": {
      "type": "string",
      "description": "Template for LLM-based evaluation"
    }
  }
}
```

### 7.4 Evaluation Specification Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://ieee.org/schemas/ai-evaluation/specification/1.0",
  "title": "AI Evaluation Specification Schema",
  "type": "object",
  "required": ["schema_version", "type", "id", "name", "datasets", "rubrics", "statistical_plan"],
  "properties": {
    "schema_version": {"type": "string", "const": "1.0"},
    "type": {"type": "string", "const": "evaluation"},
    "id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
    "name": {"type": "string", "minLength": 1, "maxLength": 255},
    "description": {"type": "string", "maxLength": 2000},
    "version": {
      "type": "string",
      "pattern": "^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)$"
    },
    "datasets": {
      "type": "array",
      "minItems": 1,
      "items": {
        "oneOf": [
          {"type": "string", "description": "Dataset ID reference"},
          {"$ref": "#/$defs/embedded_dataset"}
        ]
      }
    },
    "rubrics": {
      "type": "array",
      "minItems": 1,
      "items": {
        "oneOf": [
          {"type": "string", "description": "Rubric ID reference"},
          {"$ref": "#/$defs/embedded_rubric"}
        ]
      }
    },
    "statistical_plan": {
      "type": "object",
      "required": ["primary_metric", "significance_level"],
      "properties": {
        "primary_metric": {"type": "string"},
        "significance_level": {"type": "number", "minimum": 0.01, "maximum": 0.1, "default": 0.05},
        "multiple_comparison_correction": {
          "type": "string",
          "enum": ["bonferroni", "fdr_bh", "fdr_by", "none"],
          "default": "fdr_bh"
        },
        "confidence_interval_method": {
          "type": "string",
          "enum": ["bootstrap", "parametric", "nonparametric"],
          "default": "bootstrap"
        },
        "bootstrap_samples": {
          "type": "integer",
          "minimum": 1000,
          "default": 10000
        }
      }
    },
    "safety_requirements": {
      "type": "object",
      "properties": {
        "safety_testing": {"type": "boolean", "default": true},
        "adversarial_testing": {"type": "boolean", "default": true},
        "bias_testing": {"type": "boolean", "default": true},
        "security_assessment": {"type": "boolean", "default": false}
      }
    },
    "conformance_requirements": {
      "type": "object",
      "properties": {
        "interoperability_testing": {"type": "boolean", "default": false},
        "certification_level": {
          "type": "string",
          "enum": ["basic", "enhanced", "comprehensive"],
          "default": "basic"
        }
      }
    },
    "config": {
      "type": "object",
      "properties": {
        "max_samples": {"type": "integer", "minimum": 1},
        "randomization_seed": {"type": "integer"},
        "cross_validation": {
          "type": "object",
          "properties": {
            "method": {"type": "string", "enum": ["k_fold", "stratified", "leave_one_out"]},
            "folds": {"type": "integer", "minimum": 2, "maximum": 20, "default": 5}
          }
        },
        "model_parameters": {"type": "object"}
      }
    }
  }
}
```

### 7.5 Schema Versioning and Migration

#### 7.5.1 Version Compatibility

Schema versions shall maintain backward compatibility within major versions using semantic versioning principles:

- **Major Version:** Breaking changes requiring migration
- **Minor Version:** Backward-compatible feature additions
- **Patch Version:** Backward-compatible bug fixes

#### 7.5.2 Migration Procedures

Schema updates shall provide:

- **Automated Migration Tools:** Programmatic conversion between schema versions
- **Validation Procedures:** Verification of successful migration
- **Rollback Mechanisms:** Ability to revert to previous schema versions

## 8. Conformance and Testing

### 8.1 General

This standard establishes comprehensive conformance requirements and testing procedures ensuring interoperability, reliability, and compliance with all specified requirements.

### 8.2 Conformance Requirements

#### 8.2.1 Mandatory Requirements

Conforming implementations shall support:

- **Core Atomic Unit:** Full implementation of evaluation unit structure with input, expected output, rubric, and score
- **JSON Schema Compliance:** Validation against all specified schemas with error reporting
- **Statistical Requirements:** Confidence interval calculation, significance testing, and effect size reporting
- **Quality Assurance:** Dataset validation, inter-annotator agreement measurement, and bias assessment
- **Safety Protocols:** Basic bias detection and safety evaluation capabilities

#### 8.2.2 Optional Features

Implementations may optionally support:

- **Advanced Statistical Methods:** Bayesian analysis, non-parametric tests, advanced effect size measures
- **Multi-Modal Evaluation:** Support for image, audio, video inputs and outputs
- **Agentic AI Extensions:** Multi-step evaluation workflows and environment interaction
- **Real-Time Evaluation:** Streaming evaluation capabilities with online statistical updates
- **Enhanced Security:** Advanced adversarial testing and penetration testing capabilities

#### 8.2.3 Compliance Levels

Three compliance levels are defined:

**Basic Compliance:**
- Core atomic unit implementation
- JSON schema validation
- Statistical confidence intervals (95%)
- Basic bias detection
- Standard evaluation metrics

**Enhanced Compliance:**
- All basic compliance requirements
- Advanced statistical validation
- Comprehensive bias assessment
- Safety evaluation protocols
- Multi-modal support

**Comprehensive Compliance:**
- All enhanced compliance requirements
- Agentic AI evaluation support
- Real-time evaluation capabilities
- Advanced security assessment
- Full interoperability testing

### 8.3 Interoperability Testing

#### 8.3.1 Test Framework

Interoperability testing shall verify:

- **Schema Compatibility:** Successful parsing and validation across implementations
- **Data Exchange:** Correct interpretation of evaluation datasets and results
- **Metric Compatibility:** Consistent calculation of evaluation metrics
- **Statistical Consistency:** Agreement on confidence intervals and significance tests

#### 8.3.2 Test Procedures

Testing procedures shall include:

- **Reference Implementation Testing:** Comparison against certified reference implementations
- **Cross-Platform Validation:** Testing across different operating systems and architectures
- **Performance Benchmarking:** Evaluation of computational efficiency and scalability
- **Stress Testing:** Performance under high-volume and edge case conditions

#### 8.3.3 Interoperability Metrics

Successful interoperability requires:

- **Schema Validation:** 100% compliance with JSON schema validation
- **Metric Agreement:** Statistical agreement within 95% confidence intervals
- **Performance Standards:** Response times within acceptable bounds for evaluation complexity
- **Error Handling:** Graceful handling of malformed inputs with descriptive error messages

### 8.4 Certification Procedures

#### 8.4.1 Certification Process

Certification involves:

1. **Self-Assessment:** Implementation testing against conformance requirements
2. **Documentation Review:** Technical documentation and compliance claims verification
3. **Interoperability Testing:** Third-party validation of interoperability claims
4. **Performance Evaluation:** Assessment of computational efficiency and scalability
5. **Security Assessment:** Basic security and privacy protection verification

#### 8.4.2 Certification Levels

**Basic Certification:**
- Core functionality verification
- Schema compliance testing
- Basic interoperability validation

**Enhanced Certification:**
- All basic certification requirements
- Statistical validation verification
- Safety and bias testing validation
- Performance benchmarking

**Comprehensive Certification:**
- All enhanced certification requirements
- Advanced feature validation
- Security assessment
- Long-term reliability testing

#### 8.4.3 Certification Maintenance

Certified implementations shall:

- **Annual Review:** Yearly compliance verification and testing updates
- **Version Compatibility:** Testing against new schema versions within 90 days
- **Incident Reporting:** Mandatory reporting of security or safety issues
- **Documentation Updates:** Maintenance of current compliance documentation

## 9. Use Cases and Integration

### 9.1 General

This standard supports diverse use cases from academic research to production AI system evaluation, with particular emphasis on emerging AI capabilities and integration with existing evaluation frameworks.

### 9.2 Multi-Modal Evaluations

#### 9.2.1 Multi-Modal Framework

Multi-modal evaluations shall support:

- **Input Modalities:** Text, image, audio, video, structured data combinations
- **Output Modalities:** Generated content across multiple modalities with cross-modal evaluation
- **Cross-Modal Metrics:** Evaluation methods for inputs and outputs spanning different modalities
- **Unified Scoring:** Aggregation methods for multi-modal evaluation results

#### 9.2.2 Implementation Requirements

Multi-modal support requires:

- **Data Representation:** Standardized encoding for non-text modalities with metadata
- **Evaluation Metrics:** Modality-specific and cross-modal similarity measures
- **Statistical Validation:** Appropriate methods for multi-dimensional evaluation spaces
- **Quality Assurance:** Validation procedures adapted for multi-modal content

### 9.3 Agentic AI Systems

#### 9.3.1 Agentic Evaluation Framework

Agentic AI evaluation shall address:

- **Multi-Step Evaluation:** Assessment of planning and execution across multiple actions
- **Environment Interaction:** Evaluation of system behavior in simulated or controlled environments
- **Goal Achievement:** Measurement of task completion and optimization metrics
- **Behavioral Analysis:** Assessment of decision-making patterns and strategy effectiveness

#### 9.3.2 Agentic Evaluation Components

Extended atomic units for agentic systems include:

- **Episode Definition:** Complete interaction sequences with environment state tracking
- **Action Sequences:** Ordered sets of actions with intermediate state evaluations
- **Outcome Metrics:** Goal achievement measures with efficiency and safety considerations
- **Behavioral Patterns:** Analysis of decision-making strategies and adaptation capabilities

### 9.4 Real-Time and Streaming Evaluation

#### 9.4.1 Streaming Evaluation Framework

Real-time evaluation capabilities include:

- **Incremental Statistics:** Online confidence interval updates and significance testing
- **Adaptive Sampling:** Dynamic adjustment of evaluation intensity based on performance trends
- **Performance Monitoring:** Real-time detection of evaluation quality degradation
- **Alert Systems:** Automated notification of significant performance changes

#### 9.4.2 Implementation Considerations

Streaming implementations shall provide:

- **Computational Efficiency:** Optimized algorithms for real-time statistical updates
- **Memory Management:** Bounded memory usage with appropriate data retention policies
- **Quality Assurance:** Maintenance of statistical validity in streaming contexts
- **Scalability:** Support for high-throughput evaluation scenarios

### 9.5 Federated Evaluation Approaches

#### 9.5.1 Federated Framework

Federated evaluation supports:

- **Distributed Datasets:** Evaluation across multiple organizations without data sharing
- **Privacy Preservation:** Cryptographic techniques for privacy-preserving evaluation
- **Result Aggregation:** Statistical methods for combining distributed evaluation results
- **Trust and Verification:** Mechanisms for ensuring evaluation integrity across parties

#### 9.5.2 Security and Privacy

Federated implementations require:

- **Secure Aggregation:** Cryptographic protocols for secure result combination
- **Differential Privacy:** Privacy-preserving statistical methods with quantified privacy guarantees
- **Access Control:** Authentication and authorization for federated evaluation participants
- **Audit Trails:** Comprehensive logging for security and compliance verification

---

## Annex A (informative) Bibliography

[B1] Anthropic. *Constitutional AI: Harmlessness from AI Feedback*. arXiv:2212.08073, 2022.

[B2] Bender, E. M., et al. *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* ACM FAccT, 2021.

[B3] Bommasani, R., et al. *On the Opportunities and Risks of Foundation Models*. Stanford Institute for Human-Centered AI, 2021.

[B4] Dwork, C., & Roth, A. *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science, 2014.

[B5] IEEE Std 2857-2021. *IEEE Standard for Privacy Engineering and Risk Assessment*. IEEE Computer Society, 2021.

[B6] Liang, P., et al. *Holistic Evaluation of Language Models*. Transactions on Machine Learning Research, 2023.

[B7] Mitchell, M., et al. *Model Cards for Model Reporting*. ACM Conference on Fairness, Accountability, and Transparency, 2019.

[B8] OpenAI. *GPT-4 Technical Report*. arXiv:2303.08774, 2023.

[B9] Raschka, S. *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning*. arXiv:1811.12808, 2018.

[B10] Ribeiro, M. T., Singh, S., & Guestrin, C. *"Why Should I Trust You?" Explaining the Predictions of Any Classifier*. ACM SIGKDD, 2016.

## Annex B (informative) Example Implementation

### B.1 Basic Dataset Implementation

```python
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

@dataclass
class QualityMetrics:
    sample_size: int
    inter_annotator_agreement: Dict[str, Union[float, Dict]]
    bias_assessment: Optional[Dict] = None

@dataclass
class EvaluationExample:
    input: Union[str, Dict]
    expected_output: Optional[Union[str, List[str], Dict]] = None
    target_scores: Optional[Dict] = None
    confidence_score: Optional[float] = None
    metadata: Optional[Dict] = None

@dataclass
class EvaluationDataset:
    schema_version: str = "1.0"
    type: str = "dataset"
    id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    license: Optional[str] = None
    author: Optional[Union[str, Dict]] = None
    quality_metrics: Optional[QualityMetrics] = None
    provenance: Optional[Dict] = None
    examples: List[EvaluationExample] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.examples is None:
            self.examples = []
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
    
    def validate_quality_metrics(self) -> bool:
        if not self.quality_metrics:
            return False
        
        # Check sample size adequacy
        if self.quality_metrics.sample_size < 30:
            return False
        
        # Check inter-annotator agreement
        iaa = self.quality_metrics.inter_annotator_agreement
        if 'kappa' in iaa and iaa['kappa'] < 0.7:
            return False
        if 'icc' in iaa and iaa['icc'] < 0.8:
            return False
        
        return True

# Example usage
dataset = EvaluationDataset(
    name="Mathematical Reasoning Dataset",
    description="Dataset for evaluating mathematical problem-solving capabilities",
    quality_metrics=QualityMetrics(
        sample_size=500,
        inter_annotator_agreement={
            'kappa': 0.85,
            'confidence_interval': {
                'lower': 0.78,
                'upper': 0.92,
                'confidence_level': 0.95
            }
        }
    ),
    examples=[
        EvaluationExample(
            input="What is the derivative of x^2 + 3x + 5?",
            expected_output=["2x + 3", "2*x + 3"],
            confidence_score=0.95,
            metadata={'category': 'calculus', 'difficulty': 'easy'}
        )
    ]
)
```

### B.2 Statistical Validation Implementation

```python
import numpy as np
from scipy import stats
from typing import Tuple, List
import warnings

class StatisticalValidator:
    """Statistical validation for evaluation results."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def bootstrap_confidence_interval(self, 
                                    data: np.ndarray, 
                                    statistic_func: callable = np.mean,
                                    n_bootstrap: int = 10000) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval."""
        
        if len(data) < 30:
            warnings.warn("Sample size < 30, consider larger sample for reliable CI")
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        point_estimate = statistic_func(data)
        
        return point_estimate, ci_lower, ci_upper
    
    def effect_size_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def statistical_significance_test(self, 
                                    group1: np.ndarray, 
                                    group2: np.ndarray,
                                    test_type: str = 'ttest') -> Dict[str, float]:
        """Perform statistical significance testing."""
        
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(group1, group2)
            effect_size = self.effect_size_cohens_d(group1, group2)
        elif test_type == 'mann_whitney':
            statistic, p_value = stats.mannwhitneyu(group1, group2, 
                                                   alternative='two-sided')
            effect_size = None  # Effect size calculation for non-parametric tests
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size
        }

# Example usage
validator = StatisticalValidator(confidence_level=0.95)

# Sample evaluation scores
model_scores = np.array([0.85, 0.87, 0.82, 0.88, 0.84, 0.86, 0.83, 0.89])
baseline_scores = np.array([0.78, 0.76, 0.79, 0.77, 0.75, 0.80, 0.74, 0.78])

# Bootstrap confidence interval
mean_score, ci_lower, ci_upper = validator.bootstrap_confidence_interval(model_scores)
print(f"Mean score: {mean_score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

# Statistical significance test
test_results = validator.statistical_significance_test(model_scores, baseline_scores)
print(f"P-value: {test_results['p_value']:.4f}, Effect size: {test_results['effect_size']:.3f}")
```

## Annex C (normative) Conformance Test Procedures

### C.1 Schema Validation Tests

#### C.1.1 JSON Schema Compliance

Conforming implementations shall pass the following validation tests:

**Test C.1.1.1: Basic Schema Validation**
- Input: Valid dataset/rubric/specification JSON
- Expected: Successful validation without errors
- Implementation: Use JSON Schema validator against specified schemas

**Test C.1.1.2: Invalid Schema Rejection**
- Input: JSON with missing required fields
- Expected: Validation failure with descriptive error messages
- Implementation: Test with systematically malformed inputs

**Test C.1.1.3: Version Compatibility**
- Input: Documents with different schema versions
- Expected: Appropriate handling based on compatibility matrix
- Implementation: Test migration and validation across versions

#### C.1.2 Statistical Validation Tests

**Test C.1.2.1: Confidence Interval Calculation**
- Input: Sample evaluation scores
- Expected: 95% confidence intervals within statistical tolerance
- Implementation: Compare against reference statistical implementations

**Test C.1.2.2: Significance Testing**
- Input: Two sample groups with known statistical properties
- Expected: Correct p-values and effect size calculations
- Implementation: Test against established statistical test results

### C.2 Interoperability Tests

#### C.2.1 Data Exchange Tests

**Test C.2.1.1: Cross-Platform Dataset Exchange**
- Procedure: Export dataset from Implementation A, import to Implementation B
- Expected: Identical evaluation results within statistical tolerance
- Success Criteria: Difference in aggregate scores < 1%

**Test C.2.1.2: Rubric Portability**
- Procedure: Apply same rubric across different implementations
- Expected: Consistent scoring and confidence intervals
- Success Criteria: Statistical agreement within 95% confidence intervals

#### C.2.2 Performance Tests

**Test C.2.2.1: Scalability Assessment**
- Input: Datasets of varying sizes (100, 1K, 10K, 100K examples)
- Expected: Linear or better scaling characteristics
- Success Criteria: Processing time increases ≤ O(n log n)

**Test C.2.2.2: Memory Usage**
- Input: Large evaluation specifications
- Expected: Bounded memory usage with appropriate error handling
- Success Criteria: Memory usage < 10GB for 100K example datasets

## Annex D (informative) Statistical Methods for Evaluation

### D.1 Confidence Interval Methods

#### D.1.1 Bootstrap Methods

Bootstrap confidence intervals provide distribution-free estimates suitable for complex evaluation metrics:

**Percentile Bootstrap:**
1. Resample evaluation results with replacement B times
2. Calculate statistic of interest for each bootstrap sample
3. Use percentiles of bootstrap distribution as confidence bounds

**Bias-Corrected and Accelerated (BCa) Bootstrap:**
- Corrects for bias and skewness in bootstrap distribution
- Recommended for small samples or skewed distributions
- Provides more accurate coverage probabilities

#### D.1.2 Parametric Methods

For normally distributed evaluation metrics:

**Student's t-distribution:**
- CI = x̄ ± t(α/2,n-1) × (s/√n)
- Appropriate for sample sizes n ≥ 30 or normally distributed data
- Requires assumption verification through normality tests

### D.2 Effect Size Measures

#### D.2.1 Cohen's d
- d = (μ₁ - μ₂) / σ_pooled
- Interpretation: 0.2 (small), 0.5 (medium), 0.8 (large)
- Appropriate for comparing two groups

#### D.2.2 Eta-squared (η²)
- η² = SS_between / SS_total  
- Proportion of variance explained by group differences
- Useful for ANOVA contexts with multiple groups

### D.3 Multiple Comparison Corrections

#### D.3.1 Benjamini-Hochberg FDR Control
Recommended for multiple evaluation comparisons:

1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find largest k where p_k ≤ (k/m) × α
3. Reject hypotheses 1, 2, ..., k

#### D.3.2 Bonferroni Correction
Conservative approach: α_corrected = α / m
- Use when Type I error control is critical
- May reduce statistical power significantly

## Annex E (informative) Security Considerations Checklist

### E.1 Input Validation Security

#### E.1.1 Data Sanitization
- [ ] Input validation against malicious JSON payloads
- [ ] Size limits for evaluation datasets and examples  
- [ ] Content filtering for potentially harmful inputs
- [ ] Encoding validation and normalization

#### E.1.2 Injection Attack Prevention
- [ ] SQL injection prevention in database operations
- [ ] Command injection prevention in system calls
- [ ] Template injection prevention in rubric templates
- [ ] Script injection prevention in web interfaces

### E.2 Privacy Protection

#### E.2.1 Data Privacy
- [ ] PII detection and masking in evaluation datasets
- [ ] Encryption of sensitive evaluation data at rest
- [ ] Secure transmission protocols (TLS 1.3+)
- [ ] Access logging and audit trails

#### E.2.2 Model Privacy
- [ ] Prevention of model extraction through evaluation
- [ ] Differential privacy implementation for sensitive evaluations
- [ ] Secure aggregation for federated evaluation
- [ ] Access control for evaluation results

### E.3 System Security

#### E.3.1 Authentication and Authorization
- [ ] Multi-factor authentication for administrative access
- [ ] Role-based access control for evaluation resources
- [ ] API key management and rotation
- [ ] Session management and timeout policies

#### E.3.2 Infrastructure Security
- [ ] Regular security updates and patch management
- [ ] Network segmentation and firewall configuration
- [ ] Intrusion detection and monitoring systems
- [ ] Backup and disaster recovery procedures

## Annex F (informative) Multi-Modal and Agentic Extensions

### F.1 Multi-Modal Evaluation Extensions

#### F.1.1 Extended Atomic Unit Structure

```json
{
  "input": {
    "modalities": {
      "text": "Describe this image and generate an audio description",
      "image": {"type": "reference", "url": "dataset://image_001.jpg"},
      "metadata": {"image_type": "photograph", "resolution": "1920x1080"}
    }
  },
  "expected_output": {
    "modalities": {
      "text": "A sunset over mountains with vibrant orange and pink colors",
      "audio": {"type": "reference", "url": "dataset://audio_001.wav"},
      "metadata": {"audio_duration": 15.3, "text_length": 65}
    }
  },
  "cross_modal_alignment": {
    "text_image_consistency": 0.85,
    "text_audio_consistency": 0.92,
    "overall_coherence": 0.88
  }
}
```

#### F.1.2 Multi-Modal Metrics

**Cross-Modal Similarity Measures:**
- CLIP-based text-image similarity
- Audio-text semantic alignment using speech embeddings  
- Video-text temporal consistency metrics
- Multi-modal embedding distance measures

### F.2 Agentic AI Evaluation Extensions

#### F.2.1 Episode-Based Evaluation Structure

```json
{
  "episode": {
    "id": "agent_episode_001",
    "environment": "web_navigation_task",
    "initial_state": {"url": "https://example.com", "goal": "book_flight"},
    "action_sequence": [
      {
        "step": 1,
        "action": {"type": "click", "element": "#search-button"},
        "observation": {"page_changed": true, "new_url": "https://example.com/search"},
        "reasoning": "Clicked search to find flight options"
      }
    ],
    "final_state": {"task_completed": true, "efficiency_score": 0.87},
    "evaluation_metrics": {
      "goal_achievement": 1.0,
      "efficiency": 0.87,
      "safety_violations": 0,
      "reasoning_quality": 0.92
    }
  }
}
```

#### F.2.2 Agentic Evaluation Metrics

**Goal Achievement Measures:**
- Binary task completion indicators
- Partial credit scoring for multi-step tasks
- Time-to-completion efficiency metrics
- Resource utilization optimization scores

**Behavioral Quality Measures:**
- Decision-making consistency across similar scenarios
- Adaptation capability when facing obstacles
- Safety adherence in critical situations
- Reasoning transparency and explainability

---

This enhanced IEEE standard addresses all major shortcomings identified in the critical analysis while maintaining the practical focus needed for industry adoption. The standard now provides comprehensive technical rigor, safety considerations, statistical validation requirements, and extensibility for emerging AI capabilities.