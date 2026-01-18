# 🔍 Retrievers in LangChain

A **Retriever** is a core component in **LangChain** that fetches relevant documents from a data source in response to a user query.

Think of a retriever as a **search engine for your data**.

---

## 🔄 How a Retriever Works

```
Data Source ───▶ Retriever ───▶ Relevant Documents
                    ▲
                    │
                  Query
```

### Step-by-step Flow
1. User asks a query  
2. Query reaches the Retriever  
3. Retriever connects to the data source  
4. It scans and understands documents  
5. Returns the most relevant documents  

---

## 🧠 Simple Definition

A **Retriever** is a function that:
- Takes a **query** as input  
- Returns **multiple matching documents** as output  

It works very similar to a **search engine**.

---

## ⚙️ Key Characteristics

- All retrievers in LangChain are **Runnables**
- Retrievers can be chained with:
  - LLMs
  - Prompts
  - Output parsers
- Widely used in **RAG (Retrieval-Augmented Generation)** systems

---

# 📂 Types of Retrievers

Retrievers are classified based on **two bases**:
1. Based on **Data Source**
2. Based on **Retrieval Strategy**

---

## 1️⃣ Retrievers Based on Data Source

### 📘 1. Wikipedia Retriever

**Description**
- Takes a user query
- Sends it to the Wikipedia API
- Retrieves the most relevant articles
- Returns them as LangChain Document objects

**Example**
```
Query: Abdul Kalam
→ Wikipedia API
→ Relevant articles
→ LangChain documents
```

---

### 📦 2. Vector Store Retriever (Most Common)

**How it Works**
1. Documents stored in vector databases (FAISS, Chroma)
2. Converted into vectors using embeddings
3. Query converted into a vector
4. Semantic similarity comparison
5. Returns Top-K documents

---

### 📚 3. Archive Retriever

- Searches research papers
- Used for academic and scientific use cases

---
## 2️⃣ Retrievers Based on Retrieval Strategy

## 🧩 1. MMR (Maximal Marginal Relevance)

**Purpose**
- Reduces redundancy  
- Improves diversity  
- Selects relevant yet different documents  

### 🔍 Example

**User Query**
```
What is machine learning?
```

**Documents in Vector Store**
1. Machine learning is a subset of AI that learns from data.  
2. Machine learning algorithms learn patterns from data.  
3. Deep learning is a subset of machine learning using neural networks.  
4. Supervised learning uses labeled data to train models.  

**Normal Similarity Search Result**
```
1, 2, 3
```
❌ Redundant results

**MMR Result**
```
1, 3, 4
```
✅ Relevant and diverse information

---

## 🔁 2. Multi-Query Retriever

**Purpose**
- Handles ambiguous user queries  
- Improves recall and coverage  

### 🔍 Example

**User Query (Ambiguous)**
```
How can I stay healthy?
```

**LLM Generated Queries**
```
1. What food should I eat to stay healthy?
2. How much exercise is required for good health?
3. How does stress affect health?
4. How important is sleep for health?
```

**Process**
- Each query searches the vector store
- Documents from all searches are merged
- Duplicates are removed

**Final Result**
Documents covering nutrition, exercise, stress, and sleep

---

## 🧠 3. Contextual Compression Retriever

**Problem**
Long documents contain irrelevant information that wastes the LLM context window.

### 🔍 Example

**User Query**
```
What is photosynthesis?
```

**Retrieved Document**
```
The Grand Canyon is a famous natural tourist destination.
Photosynthesis is the process by which plants convert light energy into chemical energy.
Millions of tourists visit the Grand Canyon every year.
```

❌ Only one sentence is relevant

**Compressed Output**
```
Photosynthesis is the process by which plants convert light energy into chemical energy.
```

**Benefits**
- Reduced context size
- Lower token usage
- Higher answer accuracy

---

## 🎯 Conclusion

Choosing the right retriever strategy (MMR, Multi-Query, or Contextual Compression) significantly improves the quality and relevance of responses in RAG-based applications.
