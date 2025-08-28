**P\<designation\>™/D1**

# Draft Standard for Evaluating LLM-Powered Applications, Systems, and Products

Developed by the  
**Artificial Intelligence Standards Committee (AISC)**  
of the  
**IEEE Computer Society** 

Approved \<Date TBD\>

**IEEE SA Standards Board**

Copyright © 2025 by The Institute of Electrical and Electronics Engineers, Inc.  
Three Park Avenue  
New York, New York 10016-5997, USA

All rights reserved.

This document is an unapproved draft of a proposed IEEE Standard. As such, this document is subject to change. USE AT YOUR OWN RISK\! IEEE copyright statements SHALL NOT BE REMOVED from draft or approved IEEE standards, or modified in any way. Because this is an unapproved draft, this document must not be utilized for any conformance/compliance purposes. Permission is hereby granted for officers from each IEEE Standards Working Group or Committee to reproduce the draft document developed by that Working Group for purposes of international standardization consideration. IEEE Standards Department must be informed of the submission for consideration prior to any reproduction for international standardization consideration (stds-ipr@ieee.org). Prior to adoption of this document, in whole or in part, by another standards development organization, permission must first be obtained from the IEEE Standards Department (stds-ipr@ieee.org). When requesting permission, IEEE Standards Department will require a copy of the standard development organization’s document highlighting the use of IEEE content. Other entities seeking permission to reproduce this document, in whole or in part, must also obtain permission from the IEEE Standards Department.

IEEE Standards Department  
445 Hoes Lane  
Piscataway, NJ 08854, USA

**Abstract:** This standard specifies a structured, interoperable framework for evaluating LLM-powered applications and systems, grounded in the definition of a single model-task evaluation — the atomic unit consisting of an input, an expected output (“Golden”), a rubric, and a resulting score. JSON-based schemas define datasets, rubrics, and specifications as structured compositions of these atomic units, enabling consistent, reproducible, and comparable assessments across tools and organizations.

**Keywords:** AI evaluation, golden datasets, evaluation rubrics, evaluation specifications, JSON schema, model benchmarking, interoperability  
---

**Important Notices and Disclaimers Concerning IEEE Standards Documents**

IEEE Standards documents are made available for use subject to important notices and legal disclaimers. These notices and disclaimers, or a reference to this page [(https://standards.ieee.org/ipr/disclaimers.html)](https://standards.ieee.org/ipr/disclaimers.html), appear in all IEEE standards and may be found under the heading “Important Notices and Disclaimers Concerning IEEE Standards Documents.”

**Notice and Disclaimer of Liability Concerning the Use of IEEE Standards Documents**

IEEE Standards documents are developed within IEEE Societies and subcommittees of IEEE Standards Association (IEEE SA) Board of Governors. IEEE develops its standards through an accredited consensus development process, which brings together volunteers representing varied viewpoints and interests to achieve the final product. IEEE standards are documents developed by volunteers with scientific, academic, and industry-based expertise in technical working groups. Volunteers involved in technical working groups are not necessarily members of IEEE or IEEE SA and participate without compensation from IEEE. While IEEE administers the process and establishes rules to promote fairness in the consensus development process, IEEE does not independently evaluate, test, or verify the accuracy of any of the information or the soundness of any judgments contained in its standards.

IEEE makes no warranties or representations concerning its standards, and expressly disclaims all warranties, express or implied, concerning all standards, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. IEEE Standards documents do not guarantee safety, security, health, or environmental protection, or compliance with law, or guarantee against interference with or from other devices or networks. In addition, IEEE does not warrant or represent that the use of the material contained in its standards is free from patent infringement. IEEE Standards documents are supplied “AS IS” and “WITH ALL FAULTS.”  
Use of an IEEE standard is wholly voluntary. The existence of an IEEE standard does not imply that there are no other ways to produce, test, measure, purchase, market, or provide other goods and services related to the scope of the IEEE standard. Furthermore, the viewpoint expressed at the time a standard is approved and issued is subject to change brought about through developments in the state of the art and comments received from users of the standard.  
In publishing and making its standards available, IEEE is not suggesting or rendering professional or other services for, or on behalf of, any person or entity, nor is IEEE undertaking to perform any duty owed by any other person or entity to another. Any person utilizing any IEEE Standards document should rely upon their own independent judgment in the exercise of reasonable care in any given circumstances or, as appropriate, seek the advice of a competent professional in determining the appropriateness of a given IEEE standard.

IN NO EVENT SHALL IEEE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO: THE NEED TO PROCURE SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE PUBLICATION, USE OF, OR RELIANCE UPON ANY STANDARD, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE AND REGARDLESS OF WHETHER SUCH DAMAGE WAS FORESEEABLE.

**Translations**

The IEEE consensus balloting process involves the review of documents in English only. In the event that an IEEE standard is translated, only the English language version published by IEEE is the approved IEEE standard.

**Use by Artificial Intelligence Systems**

In no event shall material in any IEEE Standards documents be used for the purpose of creating, training, enhancing, developing, maintaining, or contributing to any artificial intelligence systems without the express, written consent of IEEE SA in advance. “Artificial intelligence” refers to any software, application, or other system that uses artificial intelligence, machine learning, or similar technologies, to analyze, train, process, or generate content. Requests for consent can be submitted using the Contact Us form.

**Official Statements**

A statement, written or oral, that is not processed in accordance with the IEEE SA Standards Board Operations Manual is not, and shall not be considered or inferred to be, the official position of IEEE or any of its committees and shall not be considered to be, or be relied upon as, a formal position of IEEE or IEEE SA. At lectures, symposia, seminars, or educational courses, an individual presenting information on IEEE standards shall make it clear that the presenter’s views should be considered the personal views of that individual rather than the formal position of IEEE, IEEE SA, the Standards Committee, or the Working Group. Statements made by volunteers may not represent the formal position of their employer(s) or affiliation(s). News releases about IEEE standards issued by entities other than IEEE SA should be considered the view of the entity issuing the release rather than the formal position of IEEE or IEEE SA.

**Comments on Standards**

Comments for revision of IEEE Standards documents are welcome from any interested party, regardless of membership affiliation with IEEE or IEEE SA. Suggestions for changes in documents should be in the form of a proposed change of text, together with appropriate supporting comments. Comments on standards should be submitted using the Contact Us form [(https://standards.ieee.org/contact/)](https://standards.ieee.org/contact/).

**Laws and Regulations**

Users of IEEE Standards documents should consult all applicable laws and regulations. Compliance with the provisions of any IEEE Standards document does not constitute compliance to any applicable regulatory requirements. Implementers of the standard are responsible for observing or referring to the applicable regulatory requirements. IEEE does not, by the publication of its standards, intend to urge action that is not in compliance with applicable laws, and these documents may not be construed as doing so.

**Data Privacy**

Users of IEEE Standards documents should evaluate the standards for considerations of data privacy and data ownership in the context of assessing and using the standards in compliance with applicable laws and regulations.

**Copyrights**

IEEE draft and approved standards are copyrighted by IEEE under U.S. and international copyright laws. They are made available by IEEE and are adopted for a wide variety of both public and private uses. These include both use by reference, in laws and regulations, and use in private self-regulation, standardization, and the promotion of engineering practices and methods. By making these documents available for use and adoption by public authorities and private users, neither IEEE nor its licensors waive any rights in copyright to the documents.  
**Photocopies**

Subject to payment of the appropriate licensing fees, IEEE will grant users a limited, non-exclusive license to photocopy portions of any individual standard for company or organizational internal use or individual, non-commercial use only. To arrange for payment of licensing fees, please contact Copyright Clearance Center, Customer Service, 222 Rosewood Drive, Danvers, MA 01923 USA; \+1 978 750 8400; [https://www.copyright.com/](https://www.copyright.com/). Permission to photocopy portions of any individual standard for educational classroom use can also be obtained through the Copyright Clearance Center.

**Updating of IEEE Standards Documents**

Users of IEEE Standards documents should be aware that these documents may be superseded at any time by the issuance of new editions or may be amended from time to time through the issuance of amendments, corrigenda, or errata. An official IEEE document at any point in time consists of the current edition of the document together with any amendments, corrigenda, or errata then in effect. Every IEEE standard is subjected to review at least every 10 years. When a document is more than 10 years old and has not undergone a revision process, it is reasonable to conclude that its contents, although still of some value, do not wholly reflect the present state of the art. Users are cautioned to check to determine that they have the latest edition of any IEEE standard. In order to determine whether a given document is the current edition and whether it has been amended through the issuance of amendments, corrigenda, or errata, visit IEEE Xplore or contact IEEE [(https://standards.ieee.org/)](https://standards.ieee.org/).

**Errata**

Errata, if any, for all IEEE standards can be accessed on the IEEE SA Website [(https://standards.ieee.org/)](https://standards.ieee.org/). Search for standard number and year of approval to access the web page of the published standard. Errata links are located under the Additional Resources Details section. Errata are also available in IEEE Xplore. Users are encouraged to periodically check for errata.

**Patents**

IEEE standards are developed in compliance with the IEEE SA Patent Policy [(https://standards.ieee.org/about/sasb/patcom/patents.html)](https://standards.ieee.org/about/sasb/patcom/patents.html). Attention is called to the possibility that implementation of this standard may require use of subject matter covered by patent rights. By publication of this standard, no position is taken by the IEEE with respect to the existence or validity of any patent rights in connection therewith. If a patent holder or patent applicant has filed a statement of assurance via an Accepted Letter of Assurance, then the statement is listed on the IEEE SA Website. Letters of Assurance may indicate whether the Submitter is willing or unwilling to grant licenses under patent rights without compensation or under reasonable rates, with reasonable terms and conditions that are demonstrably free of any unfair discrimination to applicants desiring to obtain such licenses. Essential Patent Claims may exist for which a Letter of Assurance has not been received. The IEEE is not responsible for identifying Essential Patent Claims for which a license may be required, for conducting inquiries into the legal validity or scope of Patents Claims, or determining whether any licensing terms or conditions provided in connection with submission of a Letter of Assurance, if any, or in any licensing agreements are reasonable or non-discriminatory. Users of this standard are expressly advised that determination of the validity of any patent rights, and the risk of infringement of such rights, is entirely their own responsibility. Further information may be obtained from the IEEE Standards Association.

**IMPORTANT NOTICE**

Technologies, application of technologies, and recommended procedures in various industries evolve over time. The IEEE standards development process allows participants to review developments in industries, technologies, and practices, and to determine what, if any, updates should be made to the IEEE standard. During this evolution, the technologies and recommendations in IEEE standards may be implemented in ways not foreseen during the standard’s development. IEEE standards development activities consider research and information presented to the standards development group in developing any safety recommendations. Other information about safety practices, changes in technology or technology implementation, or impact by peripheral systems also may be pertinent to safety considerations during implementation of the standard. Implementers and users of IEEE Standards documents are responsible for determining and complying with all appropriate safety, security, environmental, health, data privacy, and interference protection practices and all applicable laws and regulations.

**Participants**

At the time this draft Standard was completed, the AI Model Evaluation Working Group had the following membership:

\<Name TBD\>, Chair  
\<Name TBD\>, Vice Chair  
Participant1  
Participant2  
Participant3  
Participant4  
Participant5  
Participant6  
Participant7  
Participant8  
Participant9

The following members of the individual/entity Standards Association balloting group voted on this Standard. Balloters may have voted for approval, disapproval, or abstention.

\[To be supplied by IEEE\]  
Balloter1  
Balloter2  
Balloter3  
Balloter4  
Balloter5  
Balloter6  
Balloter7  
Balloter8  
Balloter9

When the IEEE SA Standards Board approved this Standard on \<Date TBD\>, it had the following membership:

\[To be supplied by IEEE\]  
\<Name\>, Chair  
\<Name\>, Vice Chair  
\<Name\>, Past Chair  
\<Name\>, Secretary  
SBMember1  
SBMember2  
SBMember3  
SBMember4  
SBMember5  
SBMember6  
SBMember7  
SBMember8  
SBMember9  
\*Member Emeritus  
---

**Introduction**

This introduction is not part of P\<number\>/D1, Draft Standard for Evaluating LLM-Powered Applications, Systems, and Products.  
This standard addresses the need for consistent, interoperable, and reproducible evaluation of LLM-powered applications and systems. At its foundation, the standard defines the atomic unit of evaluation: a single model-task evaluation consisting of an input, an expected output (“Golden”), a rubric, and a resulting score. Higher-level constructs — evaluation datasets, rubrics, and specifications — are structured compositions of these atomic units. By grounding the framework in this universal format, the standard ensures clarity, extensibility, and comparability across both technical benchmarking and application-level evaluations.  
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
5. JSON Schema Specifications  
   5.1 General  
   5.2 Dataset Schema  
   5.3 Rubric Schema  
   5.4 Evaluation Specification Schema  
6. Use Cases and Integration  
   6.1 General  
   6.2 Reusable Benchmarks  
   6.3 AI Product Validation  
   6.4 Cross-Team Sharing  
   6.5 Tool Integration  
   Annex A (informative) Bibliography

---

**Draft Standard for Evaluating LLM-Powered Applications, Systems, and Products**

**1\. Overview**

**1.1 Scope**

This standard specifies a framework for evaluating LLM-powered applications and systems, anchored in single model-task evaluations as the atomic unit. Each evaluation unit consists of an input, expected output (“Golden”), rubric, and resulting score. These atomic units may be composed into higher-level constructs to support product-level and system-level evaluations.

It defines JSON-based schemas for:

* **Evaluation Datasets:** Collections of atomic evaluations serving as ground truth benchmarks.  
* **Evaluation Rubrics:** Metrics and criteria for assessing model outputs against expected outputs, including exact match, similarity scores, and model-based judging.  
* **Evaluation Specifications:** Configurations that combine datasets and rubrics into executable evaluation workflows.

The standard shall ensure interoperability, shareability, and reproducibility of AI evaluations across tools and organizations. It is designed to be extensible to multi-turn or agent-based evaluations in future revisions.

**1.2 Purpose**

The purpose of this standard is to establish a common, interoperable format for AI model evaluation, replacing fragmented, ad-hoc methods with a structured, machine-readable schema. It enables AI developers, product managers, and researchers to define, share, and execute evaluation workflows, ensuring consistent and comparable assessments of model performance. The standard supports continuous evaluation, cross-team benchmarking, and integration with existing AI development tools.

**1.3 Word Usage**

The word **shall** indicates mandatory requirements strictly to be followed in order to conform to the standard and from which no deviation is permitted (shall equals is required to).

The word **should** indicates that among several possibilities, one is recommended as particularly suitable, without mentioning or excluding others; or that a certain course of action is preferred but not necessarily required (should equals is recommended that).

The word **may** is used to indicate a course of action permissible within the limits of the standard (may equals is permitted to).

The word **can** is used for statements of possibility and capability, whether material, physical, or causal (can equals is able to).

**2\. Normative References**

The following referenced documents are indispensable for the application of this document (i.e., they must be understood and used, so each referenced document is cited in text and its relationship to this document is explained). For dated references, only the edition cited applies. For undated references, the latest edition of the referenced document (including any amendments or corrigenda) applies.

* **JSON Schema Core**, Internet Engineering Task Force (IETF) Draft, draft-bhutton-json-schema-01, June 2022\.  
* **RFC 2119**, Key words for use in RFCs to Indicate Requirement Levels, S. Bradner, March 1997\.

**3\. Definitions, Acronyms, and Abbreviations**

**3.1 Definitions**

For the purposes of this document, the following terms and definitions apply. The IEEE Standards Dictionary Online should be consulted for terms not defined in this clause.

**Evaluation Dataset**: A collection of input-output pairs that serve as ground truth benchmarks for evaluating AI model performance. Synonym: Golden Dataset.

**Evaluation Rubric**: A set of criteria or metrics used to assess AI model outputs against expected outputs, such as exact match, similarity scores, or model-based judging.

**Evaluation Specification**: A configuration that defines an AI evaluation workflow by combining one or more evaluation datasets and rubrics, including execution parameters.

**Ground Truth**: The correct or ideal output for a given input, used as a reference for evaluating AI model responses.

**Evaluation Unit (Atomic Evaluation):** A single model-task evaluation consisting of an input, an expected output (Golden), a rubric, and a resulting score. All other evaluation constructs in this standard (datasets, rubrics, evaluation specifications) are structured compositions of evaluation units.

**3.2 Acronyms and Abbreviations**

AI: Artificial Intelligence  
JSON: JavaScript Object Notation  
LLM: Large Language Model

**4\. Evaluation Framework Components**

**4.1 General**

This standard defines three core components for AI model evaluation: Evaluation Datasets, Evaluation Rubrics, and Evaluation Specifications. Each component shall be represented in a JSON-based schema that is human-readable, machine-processable, and interoperable across evaluation tools.

**4.2 Evaluation Datasets**

Evaluation Datasets shall consist of a collection of examples, each containing an input (e.g., a prompt or question) and one or more expected outputs (ground truth). Datasets shall include metadata such as a unique identifier, name, description, and version.  
Each example in a dataset shall include:

* **Input**: A string or structured JSON object representing the input to the AI model.  
* **Expected Output**: A string or list of strings defining acceptable outputs, or a target\_scores object for multiple-choice evaluations.  
* **Metadata** (optional): Additional information, such as difficulty level or category tags.

Datasets shall support single-turn question-and-answer tasks and may be extended to multi-turn tasks in future revisions.

**4.3 Evaluation Rubrics**

Evaluation Rubrics shall define the criteria or metrics for assessing AI model outputs. Rubrics shall specify:

* A unique identifier, name, and description.  
* A metric type (e.g., "exact\_match", "regex\_match", "embedding\_similarity", "llm\_judge").  
* Parameters for the metric, if applicable (e.g., case sensitivity, regex pattern).  
* Score type (e.g., binary, numeric, categorical).

Rubrics may include a prompt template for model-based judging (e.g., using an LLM to evaluate outputs). Multiple rubrics may be applied to a single dataset to assess different dimensions, such as accuracy and bias.

**4.4 Evaluation Specifications**

Evaluation Specifications shall define a complete evaluation workflow by combining one or more datasets and rubrics. Each specification shall include:

* A unique identifier, name, and description.  
* References to dataset(s) and rubric(s) by ID or embedded objects.  
* Optional configuration parameters, such as maximum samples or model-specific settings.  
* Expected metrics to be reported (e.g., Accuracy, Bias Score).

Evaluation Specifications shall not include model outputs but shall serve as templates for executing evaluations.  
**4.5 Evaluation Aggregation**

Evaluation Specifications shall define how individual evaluation unit results are aggregated across the included datasets and rubrics. At minimum, rubric-level scores (e.g., accuracy, bias) shall be reported. Implementers may optionally define composite or application-level scores by applying weights or multi-dimensional indices across rubrics. Such aggregation methods shall be documented within the Evaluation Specification to ensure reproducibility and comparability.

This standard does not prescribe a specific aggregation formula; Annex B provides an informative example.

**5\. JSON Schema Specifications**

**5.1 General**

The JSON schemas for Evaluation Datasets, Rubrics, and Specifications shall conform to the JSON Schema Core specification. Each schema shall include a "schema\_version" field set to "1.0" for this standard.

**5.2 Dataset Schema**

The Dataset schema shall include the following fields:

* **schema\_version** (string, required): Version of the standard (e.g., "1.0").  
* **type** (string, required): Set to "dataset".  
* **id** (string, required): Unique identifier (e.g., "math-arith-v1").  
* **name** (string, required): Human-readable name.  
* **description** (string, optional): Purpose of the dataset.  
* **version** (string, optional): Dataset content version.  
* **license** (string, optional): License identifier (e.g., "CC-BY-SA-4.0").  
* **author** (object or string, optional): Creator information.  
* **examples** (array of objects, required): List of evaluation cases, each with:  
  * **input** (required): Input prompt or question.  
  * **expected\_output** (required unless using target\_scores): Ground truth answer(s).  
  * **target\_scores** (optional): Scores for multiple-choice options.  
  * **metadata** (object, optional): Additional example information.

**Example**:  
json  
{  
  "schema\_version": "1.0",  
  "type": "dataset",  
  "id": "math\_arith\_v1",  
  "name": "Basic Arithmetic QA",  
  "description": "Simple addition and subtraction questions.",  
  "version": "1.0",  
  "license": "CC-BY-SA-4.0",  
  "author": { "name": "Alice Example", "contact": "alice@example.com" },  
  "examples": \[  
    {  
      "input": "1 \+ 1 \= ?",  
      "expected\_output": \["2", "two"\],  
      "metadata": { "category": "addition", "difficulty": "easy" }  
    },  
    {  
      "input": "5 \- 3 \= ?",  
      "expected\_output": "2",  
      "metadata": { "category": "subtraction", "difficulty": "easy" }  
    }  
  \]  
}

**5.3 Rubric Schema**

The Rubric schema shall include:

* **schema\_version**, **type**, **id**, **name**, **description**, **version**, **license**: As defined in 5.2.  
* **metric** (string, required): Metric type (e.g., "exact\_match", "llm\_judge").  
* **params** (object, optional): Metric-specific parameters.  
* **score\_type** (string, optional): Score format (e.g., "binary").  
* **prompt\_template** (string, optional): Template for LLM-based judging.

**Example**:  
json  
{  
  "schema\_version": "1.0",  
  "type": "rubric",  
  "id": "exact\_match",  
  "name": "Exact Match",  
  "description": "Checks if the model output exactly matches the expected output string (case-insensitive, trimming whitespace).",  
  "metric": "exact\_match",  
  "params": { "case\_sensitive": false, "trim\_whitespace": true },  
  "score\_type": "binary"  
}

**5.4 Evaluation Specification Schema**

The Evaluation Specification schema shall include:

* **schema\_version**, **type**, **id**, **name**, **description**, **version**: As defined in 5.2.  
* **dataset\_id** (string) or **dataset** (object): Reference to or embedded dataset.  
* **rubric\_ids** (array of strings) or **rubrics** (array of objects): References to or embedded rubrics.  
* **metrics** (array of strings, optional): Reported metrics.  
* **primary\_metric** (string, optional): Primary metric for focus.  
* **config** (object, optional): Execution parameters.

**Example**:  
json  
{  
  "schema\_version": "1.0",  
  "type": "evaluation",  
  "id": "eval\_basic\_math\_v1",  
  "name": "Basic Math Skills Evaluation",  
  "description": "Evaluates basic addition and subtraction using an exact match rubric.",  
  "dataset\_id": "math\_arith\_v1",  
  "rubric\_ids": \["exact\_match"\],  
  "metrics": \["Accuracy"\],  
  "primary\_metric": "Accuracy",  
  "config": { "max\_samples": 3, "model\_parameters": { "temperature": 0.0 } }  
}

**6\. Use Cases and Integration**

**6.1 General**

This standard shall support use cases such as reusable benchmarks, AI product validation, cross-team sharing, and tool integration. Implementations should ensure compatibility with existing AI evaluation frameworks.

**6.2 Reusable Benchmarks**

Evaluation Datasets, Rubrics, and Specifications shall be reusable across model versions to enable consistent performance comparisons.

**6.3 AI Product Validation**

The standard shall enable semi-technical stakeholders (e.g., product managers) to define evaluation criteria in human-readable JSON, facilitating alignment between product goals and technical metrics.

**6.4 Cross-Team Sharing**

The JSON-based schemas shall enable sharing of evaluation assets across teams and organizations, supporting standardized benchmarking.

**6.5 Tool Integration**

Implementations may integrate with tools such as OpenAI Evals, LangChain/LangSmith, or custom MLOps pipelines by mapping JSON schemas to tool-specific formats.

**Annex A** 

**(informative)**

**Bibliography**

\[B1\] OpenAI Evals Framework, GitHub, [https://github.com/openai/evals](https://github.com/openai/evals).  
\[B2\] BIG-bench JSON Task Schema, GitHub, [https://github.com/google/BIG-bench](https://github.com/google/BIG-bench).  
\[B3\] Anthropic Model Context Protocol, [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol).  
\[B4\] AI Evaluation Best Practices, Medium, [https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5](https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5).  
\[B5\] Confident AI’s Evaluation Guide, [https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation).

### **Annex B (informative)**

**Example of Application-Level Aggregation**

**Note B.1 — Aggregating evaluation results at the application level**  
While Clause 4.5 defines the requirement that rubric-level scores *shall* be reported and that composite scores *may* be defined, this annex provides an illustrative example of how an application-level score can be derived.

Consider an AI-powered customer support application evaluated with three rubrics:

* **Accuracy** (binary exact match of expected answers).  
* **Bias Sensitivity** (LLM-judge rubric to detect biased language).  
* **Fluency** (embedding similarity rubric to measure naturalness of responses).

Each evaluation unit produces results for these rubrics. The dataset yields:

* Accuracy: 85%  
* Bias Sensitivity: 92%  
* Fluency: 78%

A team may define an **application-level score** in the Evaluation Specification by applying weights that reflect business priorities, for example:

* Accuracy: 50%  
* Bias Sensitivity: 30%  
* Fluency: 20%

The composite score is then:  
(0.85 × 0.5) \+ (0.92 × 0.3) \+ (0.78 × 0.2) \= **0.854 (85.4%)**

This aggregated score provides a single comparative metric for decision-makers, while still preserving rubric-level detail for diagnostic purposes.

**Important:** The standard does not mandate a specific formula for aggregation. Implementers *shall* document chosen aggregation methods within the Evaluation Specification to ensure reproducibility and comparability.

---

**Notes on Refactoring and IEEE SA Style Manual Compliance**

1. **Structure**: The content is reorganized into IEEE’s mandatory sections (Scope, Purpose, Word Usage, Normative References, Definitions, etc.), with technical details in clauses 4 and 5, and use cases in clause 6, per IEEE SA Style Manual guidelines.  
2. **Word Usage**: Mandatory requirements use "shall," recommendations use "should," and permissive actions use "may," as defined in IEEE SA Standards Board Operations Manual 6.4.7.  
3. **Normative References**: Included JSON Schema Core and RFC 2119, as they are critical for schema validation and requirement terminology.  
4. **Licensing**: The Evalsify doc’s CC-BY-SA-4.0 and MIT licenses are noted but not included in the IEEE doc, as IEEE standards are copyrighted by IEEE. Users would need to address licensing separately for shared assets.  
5. **Condensed Content**: Examples and use cases are streamlined to fit the IEEE format while retaining core concepts. Detailed how-to guides are omitted, as they are more suited to informative annexes or external documentation.  
6. **Placeholders**: Participant and balloter lists are left as placeholders, as they would be populated during the IEEE standards process.

If you need further refinements, specific sections expanded, or additional IEEE-style content (e.g., more definitions or normative clauses), please let me know\!

