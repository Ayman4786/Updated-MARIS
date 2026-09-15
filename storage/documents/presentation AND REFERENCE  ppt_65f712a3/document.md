

<!-- PAGE 1 -->

<!-- image -->

## The Oxford College of Engineering

## DEPARTMENT OF ARTIFICIAL INTELLIGENCE &amp; MACHINE LEARNING

## PHASE 2 - FIRST REVIEW MULTIMODAL DOCUMENT QA ASSISTANT WITH GROUNDED REASONING &amp; HIGHLIGHTING

Team Member's:

1. BHANU PRAKASH V S
2. AYMAN KHAN
3. BHAVESH REDDY
4. GAGAN S

Guide: SAVITHA H P

Assistant Professor

Dept. of AIML

<!-- PAGE 2 -->

## PROBLEM STATEMENT

<!-- image -->

<!-- image -->

- [x] Existing RAG systems mainly depend on extracted text from documents .

- [x] Text extraction / OCR can lose or distort layout, tables, diagrams, and visual information .

- [x] There is need of Vision based multimodal RAG approach capable of retrieving and reasoning over the original documents .

Modern documents contain texts , image &amp; tables .

<!-- PAGE 3 -->

## MOTIVATION

## Future of AI:

Exploring the transition from text -based AI to multimodal vision -enabled AI systems .

Frontier Technology: Gaining hands onexperience with VLMs, Multimodal retrieval, embeddings and Modern RAG architectures .

## Research &amp; Innovation:

Building on recent research such as VisRAG and following a research driven approach rather than simple application .

Real World Impact: Applying modern AI to modern documents where accurate information retrieval is valuable .

<!-- PAGE 4 -->

## OBJECTIVES

<!-- image -->

## Study &amp; Understand VisRAG:

Analyze the VisRAG architecture and its core components. Understand vision-based retrieval and generation.

<!-- image -->

## Develop a Multimodal RAG Pipeline:

Build a vision-based RAG pipeline for maritime technical documents. Explore VisionLanguage Models, visual embeddings, and retrieval.

<!-- image -->

## Evaluate on Modern Documents:

Test the approach on modern manuals, tables, diagrams, and technical content.  Evaluate retrieval and answergeneration performance.

<!-- image -->

## Identify &amp; Improve Limitations:

Analyze system strengths, limitations, and failure cases. Investigate domain-specific improvements for MARIS.

<!-- PAGE 5 -->

## EXISTING SYSTEMS

## Multimodal Retrieval-Augmented Generation(RAG) Architecture forAcademicReporting

Unstructured academicandstudydocuments(PDFs,handwrittennotes,/textbooks)→groundedLLMresponsesinverifiedsourceevidence. Thesystemretrieves,augments,andgeneratesaccurate,traceable,andcontext-aware academicanswers.

<!-- image -->

## KeyImplementationHighlights

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

## GroundedGeneration

LLManswers aregrounded inretrievedevidenceto ensurefactual accuracy.

<!-- image -->

<!-- PAGE 6 -->

## LITERATURE SURVEY

|   S.No. | Paper                                                                                      | Authors                 |   Year | Advantage                                                                                    | Research Gap                                                                            | Result                                                |
|---------|--------------------------------------------------------------------------------------------|-------------------------|--------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-------------------------------------------------------|
|       1 | VisRAG:Vision-based Retrieval- Augmented Generation on Multi- ModalityDocuments            | Shi Yu et al.           |   2023 | Direct visual retrieval from document pages; preserves layouts, tables and figures           | Generic-domain approach; limited maritime-specific evaluation                           | Strong visual RAG baseline; selectedasMARiSbase paper |
|       2 | ColPali:EfficientDocumentRetrieval withVisionLanguageModels                                | Manuel Faysse etal.     |   2024 | Effectivevisual documentretrieval usingmulti-vector representations                          | Higherstorage/compute; mainlyretrieval-focused                                          | Improvesvisualretrieval quality                       |
|       3 | M3DocRAG:Multi-modalRetrievalis What You Need for Multi-page Multi- document Understanding | Jaemin Cho et al.       |   2023 | Handles multi-page and multi- document multimodal understanding                              | More complex pipeline; domain-specific reasoning remains open                           | Improvesmulti-document reasoning and retrieval        |
|       4 | Vision-Guided Chunking Is All You Need                                                     | Vishesh Tripathi et al. |   2022 | Preserves document structure, tables and figures during chunking technical-domain validation | Computational overhead;needsEnables structure-aware                                     | retrieval                                             |
|       5 | Rewrite-Retrieve-Read                                                                      | Xinbei Ma et al.        |   2023 | Rewrites queries to improve retrieval                                                        | Not specifically designed for visual/maritime queries                                   | Improvesretrievalfor ambiguous queries                |
|       6 | Adaptive-RAG                                                                               | Soyeong Jeong et al.    |   2024 | Selects retrieval strategy accordingtoquerycomplexity                                        | Mainly text-based;multimodal adaptation needed                                          | Improves retrieval efficiency and adaptability        |
|       7 | Self-RAG:LearningtoRetrieve, Generate,and Critique through Self-Reflection                 | AkariAsaiet al.         |   2023 | Self-reflection improves factuality andretrieval decisions                                   | Additional inference complexity;visual evidence checkingislimited                       | Improvesanswerreliability andfactuality               |
|       8 | Corrective Retrieval Augmented Generation (CRAG)                                           | Shi-Qi Yan et al.       |   2024 | Detects poor retrieval and performs correctiveretrieval                                      | Visual-page correction remains underexplored                                            | Reduces errors caused by poorretrieval                |
|       9 | Enabling Large LanguageModels to GenerateTextwith Citations(ALCE)                          | Tianyu Gao et al.       |   2023 | Improvescitationquality and answer verifiability                                             | Mostly text-based citations; visual evidence grounding remains open                     | Producesmoreverifiable answers                        |
|      10 | RAG-Fusion:a New Take on Retrieval-Augmented Generation                                    | Zackary Rackauckas      |   2024 | Multiple query perspectives andresultfusion improve retrieval coverage                       | Can introduce irrelevant results;Improves retrieval coverage not visual-domain specific | and robustness                                        |

<!-- PAGE 7 -->

## SYSTEM ARCHITECTURE

<!-- image -->

<!-- PAGE 8 -->

## FEATURES IMPLEMENTED

## 1. Project Development - 60% Progress Modules Completed

- MARIS system architecture and overall workflow finalized.
- Voice-based interaction system successfully implemented.
- AI tutor conversation flow and interaction design established.
- Document/data ingestion pipeline partially completed.
- Required tools, technologies, models, and resources finalized.

## Features Implemented

- Voice input and voice output.
- AI-powered conversational tutoring.
- Core system architecture for integrating ingestion, retrieval, reasoning, and response generation.
- Initial document ingestion capability.
- Foundation for multimodal educational assistance.

## Current Working Demo

- Functional voice-based MARIS prototype.
- User can interact with MARIS through natural voice conversation.
- System architecture is ready for integration of the remaining educational intelligence modules.

<!-- PAGE 9 -->



<!-- PAGE 10 -->



<!-- PAGE 11 -->



<!-- PAGE 12 -->



<!-- PAGE 13 -->



<!-- PAGE 14 -->



<!-- PAGE 15 -->

